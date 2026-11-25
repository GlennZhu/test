#!/usr/bin/env python3
"""Standalone replay from ./raw/*.csv (requires only Python 3.10+ and numpy)."""

from __future__ import annotations

import bisect
import csv
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
BACKTEST = ROOT / "backtest"

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
WEEK_MS = 7 * DAY_MS
SHIFT_16Z_MS = 8 * HOUR_MS
DIETZ_WEEK_EPOCH_MS = 0  # 1970-01-01T00:00:00Z (Thursday 00:00Z, immediately after Wednesday allTime points)
MIN_DIETZ_WEEK_SPAN_MS = int(6.5 * DAY_MS)
FUNDING_SETTLEMENT_JITTER_MS = 120_000
SETTLEMENT_INDEXING_GRACE_MS = 4 * 60_000

HLP_VAULT_ADDR = "0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00"
MIN_CAPITAL_USD = 3_000_000.0
MIN_HISTORY_DAYS = 30.0
MIN_DIETZ_WEEKS = 4
MIN_DIETZ_PF = 1.20
MIN_CHURN_EVAL_DAYS = 20
MAX_DAILY_CHURN = 1.0
MAX_DAILY_FLIP_RATE = 0.20
HEDGE_MIN_OFFSET_RATIO = 0.70
COMPLETE_LIFE_POS_TOLERANCE = 1e-2
CHURN_WARMUP_START_DATE = "2025-07-01"
MIN_SIGNAL_WARMUP_DATE = "2025-07-30"
FIRST_LIVE_16Z_BAR = "2026-11-13"
MIN_QUORUM_WHALES = 15

ENTRY_TILT = 0.15
EXIT_TILT = 0.05
VOL_TARGET_NUM = 0.80
MIN_LEV = 1.0
MAX_LEV = 2.0
DISASTER_STOP_PCT = 0.18
FEE_BPS = 10.0
LONG_FUNDING_BPS_DAY = 3.0
INITIAL_CAPITAL_USD = 100_000.0

FOOTPRINT_MIN_GROSS_BTC = 5000.0
FOOTPRINT_MIN_HL_OI_SHARE = 0.15
FOOTPRINT_MIN_ACTIVE_WHALES = 70
FOOTPRINT_STREAK_WARN_DAYS = 5
DECAY_IC_FORWARD_DAYS = 5
DECAY_IC_WINDOW_DAYS = 120
DECAY_MIN_ROLL120_IC = -0.05
DECAY_MAX_1X_DD_PCT = -18.0
DECAY_PAUSE_IC = -0.15
DECAY_PAUSE_1X_DD_PCT = -25.0
DECAY_CUT_SIZE_MULT = 0.50


def bar_date_for_16z_open_ms(open_ts_ms: int) -> str:
    shifted_sec = (int(open_ts_ms) + SHIFT_16Z_MS) / 1000.0
    return datetime.fromtimestamp(shifted_sec, tz=UTC).strftime("%Y-%m-%d")


def close_ts_ms_for_16z_bar(bar_date: str) -> int:
    dt = datetime.strptime(bar_date, "%Y-%m-%d").replace(hour=16, minute=0, second=0, tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def next_bar_date(bar_date: str) -> str:
    dt = datetime.strptime(bar_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def prev_bar_date(bar_date: str) -> str:
    dt = datetime.strptime(bar_date, "%Y-%m-%d").replace(tzinfo=UTC) - timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def clamp_exact_16z_since_bar(
    raw_exact_since: str,
    *,
    max_comp_bar: str = "",
    min_exact_bar: str = "",
    funding_synced_to_ms: int | None = None,
    first_live_bar: str = FIRST_LIVE_16Z_BAR,
) -> str:
    if not raw_exact_since:
        return ""
    after_comp = next_bar_date(max_comp_bar) if max_comp_bar else ""
    resolved = raw_exact_since
    if first_live_bar and resolved < first_live_bar:
        if min_exact_bar:
            floor_bar = max(after_comp, min(min_exact_bar, first_live_bar))
        else:
            floor_bar = max(after_comp, first_live_bar) if after_comp else first_live_bar
        if resolved < floor_bar:
            resolved = floor_bar
    if after_comp and resolved < after_comp:
        resolved = after_comp
    if (
        funding_synced_to_ms is not None
        and not min_exact_bar
        and resolved
        and funding_synced_to_ms < close_ts_ms_for_16z_bar(resolved)
    ):
        return ""
    return resolved


def _utc_day_bounds_ms(bar_date: str) -> tuple[int, int]:
    dt = datetime.strptime(bar_date, "%Y-%m-%d").replace(tzinfo=UTC)
    start_ms = int(dt.timestamp() * 1000)
    return start_ms, start_ms + DAY_MS


def is_ever_on_leaderboard(w: dict[str, Any]) -> bool:
    if int(w.get("on_leaderboard") or 0) == 1:
        return True
    raw_ever = w.get("ever_on_leaderboard")
    if raw_ever is not None and str(raw_ever).strip() != "":
        return int(raw_ever) == 1
    return False


def has_prefill_nonzero_funding(
    *,
    earliest_fill_ts_ms: int | None,
    funding_rows: list[tuple[str, int, float, int]],
    synced_to_ms: int = 0,
) -> bool:
    earliest_utc_date = (
        datetime.fromtimestamp(earliest_fill_ts_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")
        if earliest_fill_ts_ms is not None
        else None
    )
    for bdate, is_exact, szi, settled_ts_ms in funding_rows:
        if abs(szi) < COMPLETE_LIFE_POS_TOLERANCE:
            continue
        if is_exact == 1:
            bound_ts = (
                settled_ts_ms
                if settled_ts_ms > 0
                else (close_ts_ms_for_16z_bar(bdate) + FUNDING_SETTLEMENT_JITTER_MS)
            )
            if synced_to_ms > 0 and bound_ts > synced_to_ms:
                continue
            if earliest_fill_ts_ms is None or bound_ts < earliest_fill_ts_ms:
                return True
        else:
            if synced_to_ms > 0 and (_utc_day_bounds_ms(bdate)[1] + FUNDING_SETTLEMENT_JITTER_MS) > synced_to_ms:
                continue
            if earliest_utc_date is None or bdate < earliest_utc_date:
                return True
    return False


def fill_chain_matches_exact_funding(
    *,
    ts_list: list[int],
    ep_list: list[float],
    fills_synced_to_ms: int,
    exact_rows: list[tuple[str, float, int]],
) -> bool:
    if fills_synced_to_ms <= 0:
        return True
    for bdate, szi, settled_ts_ms in exact_rows:
        bdate_close_ms = close_ts_ms_for_16z_bar(bdate)
        if settled_ts_ms > 0:
            if settled_ts_ms > fills_synced_to_ms:
                continue
            lo = bisect.bisect_left(ts_list, settled_ts_ms) - 1
            hi = bisect.bisect_right(ts_list, settled_ts_ms) - 1
        else:
            if (bdate_close_ms + FUNDING_SETTLEMENT_JITTER_MS) > fills_synced_to_ms:
                continue
            lo = bisect.bisect_left(ts_list, bdate_close_ms - FUNDING_SETTLEMENT_JITTER_MS) - 1
            hi = bisect.bisect_right(ts_list, bdate_close_ms + FUNDING_SETTLEMENT_JITTER_MS) - 1
        matched = False
        for idx in range(lo, hi + 1):
            cand_pos = ep_list[idx] if idx >= 0 else 0.0
            if abs(cand_pos - szi) < COMPLETE_LIFE_POS_TOLERANCE:
                matched = True
                break
        if not matched:
            return False
    return True


def fill_chain_matches_compacted_funding(
    *,
    ts_list: list[int],
    ep_list: list[float],
    fills_synced_to_ms: int,
    compacted_rows: list[tuple[str, float]],
) -> bool:
    if fills_synced_to_ms <= 0:
        return True
    for bdate, szi in compacted_rows:
        day_start_ms, day_end_ms = _utc_day_bounds_ms(bdate)
        if (day_end_ms + FUNDING_SETTLEMENT_JITTER_MS) > fills_synced_to_ms:
            continue
        lo_idx = bisect.bisect_left(ts_list, day_start_ms - FUNDING_SETTLEMENT_JITTER_MS) - 1
        hi_idx = bisect.bisect_right(ts_list, day_end_ms + FUNDING_SETTLEMENT_JITTER_MS) - 1
        init_pos = ep_list[lo_idx] if lo_idx >= 0 else 0.0
        min_pos = init_pos
        max_pos = init_pos
        for idx in range(max(0, lo_idx + 1), hi_idx + 1):
            pos_val = ep_list[idx]
            if pos_val < min_pos:
                min_pos = pos_val
            if pos_val > max_pos:
                max_pos = pos_val
        if (
            szi < (min_pos - COMPLETE_LIFE_POS_TOLERANCE)
            or szi > (max_pos + COMPLETE_LIFE_POS_TOLERANCE)
        ):
            return False
    return True


def _resolve_complete_fills_until_ms(
    w: dict[str, Any],
    *,
    fill_pair: tuple[list[int], list[float]] | None,
    chain_break_ms: int | None,
    w_frows: list[tuple[str, int, float, int]],
    now_ms: int,
) -> int:
    if chain_break_ms == 0:
        return 0
    fills_synced_to_ms = int(w.get("fills_synced_to_ms") or 0)
    stored_until = int(w.get("complete_fills_until_ms") or 0)
    raw_cl = bool(int(w.get("is_complete_lifetime") or 0))
    cur_szi = float(w.get("cur_btc_szi") or 0.0)
    ts_all = fill_pair[0] if fill_pair is not None else []
    ep_all = fill_pair[1] if fill_pair is not None else []
    exact_rows = [(bd, szi, sts) for bd, is_ex, szi, sts in w_frows if is_ex == 1]
    compacted_rows = [(bd, szi) for bd, is_ex, szi, _sts in w_frows if is_ex == 0]

    if raw_cl and chain_break_ms is None:
        earliest_ts = ts_all[0] if ts_all else None
        last_ep = float(ep_all[-1]) if ep_all else 0.0
        full_valid = (
            abs(last_ep - cur_szi) < COMPLETE_LIFE_POS_TOLERANCE
            and not has_prefill_nonzero_funding(
                earliest_fill_ts_ms=earliest_ts,
                funding_rows=w_frows,
                synced_to_ms=fills_synced_to_ms,
            )
            and fill_chain_matches_exact_funding(
                ts_list=ts_all,
                ep_list=ep_all,
                fills_synced_to_ms=fills_synced_to_ms,
                exact_rows=exact_rows,
            )
            and fill_chain_matches_compacted_funding(
                ts_list=ts_all,
                ep_list=ep_all,
                fills_synced_to_ms=fills_synced_to_ms,
                compacted_rows=compacted_rows,
            )
        )
        if full_valid:
            cand_until = max(stored_until, fills_synced_to_ms)
            return min(cand_until, now_ms) if now_ms > 0 else cand_until

    if stored_until <= 0:
        return 0
    prefix_until = (
        min(stored_until, chain_break_ms - 1)
        if (chain_break_ms is not None and chain_break_ms > 0)
        else stored_until
    )
    if prefix_until <= 0:
        return 0
    idx_p = bisect.bisect_right(ts_all, prefix_until)
    ts_p = ts_all[:idx_p]
    ep_p = ep_all[:idx_p]
    earliest_p = ts_p[0] if ts_p else None
    if (
        has_prefill_nonzero_funding(
            earliest_fill_ts_ms=earliest_p,
            funding_rows=w_frows,
            synced_to_ms=prefix_until,
        )
        or not fill_chain_matches_exact_funding(
            ts_list=ts_p,
            ep_list=ep_p,
            fills_synced_to_ms=prefix_until,
            exact_rows=exact_rows,
        )
        or not fill_chain_matches_compacted_funding(
            ts_list=ts_p,
            ep_list=ep_p,
            fills_synced_to_ms=prefix_until,
            compacted_rows=compacted_rows,
        )
    ):
        return 0
    return min(prefix_until, now_ms) if now_ms > 0 else prefix_until


def build_causal_nav_features(
    raw_av_by_addr: dict[str, list[tuple[int, float, float]]],
    *,
    now_ms: int | None = None,
) -> dict[str, tuple[list[int], list[float], list[float], list[int]]]:
    nav_feat: dict[str, tuple[list[int], list[float], list[float], list[int]]] = {}
    for addr, pts in raw_av_by_addr.items():
        scoped_pts = (
            [p for p in pts if int(p[0]) <= int(now_ms)]
            if now_ms is not None
            else list(pts)
        )
        if not scoped_pts:
            continue
        pts_ts = [int(p[0]) for p in scoped_pts]
        first_ts = pts_ts[0]
        last_ts = pts_ts[-1]
        horizon_ts = max(last_ts, int(now_ms)) if (now_ms is not None and int(now_ms) > last_ts) else last_ts
        k_start = ((first_ts - DIETZ_WEEK_EPOCH_MS) // WEEK_MS) + 1
        k_end = (horizon_ts - DIETZ_WEEK_EPOCH_MS) // WEEK_MS

        events: list[tuple[int, int, float, float, int]] = [
            (int(p[0]), 0, float(p[1]), float(p[2]), int(p[0])) for p in scoped_pts
        ]
        for k in range(k_start, k_end + 1):
            w_ms = DIETZ_WEEK_EPOCH_MS + k * WEEK_MS
            idx = bisect.bisect_right(pts_ts, w_ms) - 1
            if idx >= 0:
                events.append((w_ms, 1, float(scoped_pts[idx][1]), float(scoped_pts[idx][2]), pts_ts[idx]))
        events.sort(key=lambda item: (item[0], item[1]))

        av_timestamps: list[int] = []
        peak_list: list[float] = []
        pf_list: list[float] = []
        weeks_list: list[int] = []
        peak_av = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        active_weeks = 0
        base_ts = first_ts
        base_pt_ts = first_ts
        base_av, base_cp = float(scoped_pts[0][1]), float(scoped_pts[0][2])
        prev_peak = max(base_av, 0.0)
        has_closed_week = False

        for ts_ms, is_week_close, av_val, cum_pnl, src_pt_ts in events:
            if av_val > peak_av:
                peak_av = av_val
            if is_week_close == 1:
                should_close = (
                    (src_pt_ts - base_pt_ts) >= MIN_DIETZ_WEEK_SPAN_MS
                    if not has_closed_week
                    else (src_pt_ts > base_ts and (ts_ms - base_ts) >= MIN_DIETZ_WEEK_SPAN_MS)
                )
                if should_close:
                    dp = cum_pnl - base_cp
                    net_cash_flow = (av_val - base_av) - dp
                    denom = max(base_av + 0.5 * max(net_cash_flow, 0.0), prev_peak * 0.15, 100_000.0)
                    week_ret = max(-0.80, min(2.0, dp / denom))
                    if abs(week_ret) > 1e-4:
                        active_weeks += 1
                        if week_ret > 0.0:
                            gross_profit += week_ret
                        else:
                            gross_loss += -week_ret
                    base_ts, base_pt_ts, base_av, base_cp = ts_ms, src_pt_ts, av_val, cum_pnl
                    prev_peak = peak_av
                    has_closed_week = True
            pf = (gross_profit / gross_loss) if gross_loss > 1e-9 else (99.0 if gross_profit > 0.0 else 0.0)
            if av_timestamps and av_timestamps[-1] == ts_ms:
                peak_list[-1] = peak_av
                pf_list[-1] = pf
                weeks_list[-1] = active_weeks
            else:
                av_timestamps.append(ts_ms)
                peak_list.append(peak_av)
                pf_list.append(pf)
                weeks_list.append(active_weeks)
        nav_feat[addr] = (av_timestamps, peak_list, pf_list, weeks_list)
    return nav_feat


def load_raw_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    meta = json.loads((RAW / "service_meta.json").read_text(encoding="utf-8"))
    with (RAW / "btc_candles_16z.csv").open(encoding="utf-8") as f:
        candles = []
        for r in csv.DictReader(f):
            raw_rate = (r.get("funding_rate_16z") or "").strip()
            candles.append({
                "bar_date": r["bar_date"],
                "open_ts_ms": int(r["open_ts_ms"]),
                "close_ts_ms": int(r["close_ts_ms"]),
                "btc_open": float(r["open"]),
                "btc_high": float(r["high"]),
                "btc_low": float(r["low"]),
                "btc_close": float(r["close"]),
                "btc_volume": float(r["volume"]),
                "is_complete": int(r["is_complete"]),
                "funding_rate_16z": float(raw_rate) if raw_rate != "" else None,
            })
    oi_map: dict[str, float] = {}
    with (RAW / "btc_exchange_oi_daily.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            oi_map[r["bar_date"]] = float(r["hl_btc_oi"])
    with (RAW / "wallets.csv").open(encoding="utf-8") as f:
        wallets = list(csv.DictReader(f))
    return candles, wallets, {"meta": meta, "oi_map": oi_map}


def reconstruct_signals(candles: list[dict[str, Any]], wallets: list[dict[str, Any]], now_ms: int) -> dict[str, dict[str, Any]]:
    raw_av: dict[str, list[tuple[int, float, float]]] = {}
    with (RAW / "wallet_av_history.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw_av.setdefault(r["address"], []).append((int(r["ts_ms"]), float(r["acct_val"]), float(r["cum_pnl"])))
    nav_feat = build_causal_nav_features(raw_av, now_ms=now_ms)

    fills_by_w: dict[str, tuple[list[int], list[float]]] = {}
    fill_chain_break_ms_by_w: dict[str, int] = {}
    prev_ep_by_w: dict[str, float] = {}
    with (RAW / "wallet_btc_fills.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["address"]
            if a in fill_chain_break_ms_by_w:
                continue
            ts_val = int(r["ts_ms"])
            sp = float(r["start_pos"])
            ep = float(r["end_pos"])
            if a not in prev_ep_by_w:
                if abs(sp) >= COMPLETE_LIFE_POS_TOLERANCE:
                    fill_chain_break_ms_by_w[a] = 0
                    continue
            elif abs(sp - prev_ep_by_w[a]) >= COMPLETE_LIFE_POS_TOLERANCE:
                fill_chain_break_ms_by_w[a] = max(1, ts_val)
                continue
            prev_ep_by_w[a] = ep
            pair = fills_by_w.get(a)
            if pair is None:
                pair = ([], [])
                fills_by_w[a] = pair
            pair[0].append(ts_val)
            pair[1].append(ep)

    f_exact: dict[str, dict[str, float]] = {}
    f_exact_ts: dict[str, dict[str, int]] = {}
    f_comp: dict[str, dict[str, float]] = {}
    f_rows_by_w: dict[str, list[tuple[str, int, float, int]]] = {}
    with (RAW / "wallet_daily_funding_szi.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["address"]
            bd = r["bar_date"]
            is_ex = int(r["is_exact_16z"])
            szi = float(r["szi"])
            sts = int(r.get("settled_ts_ms") or 0)
            f_rows_by_w.setdefault(a, []).append((bd, is_ex, szi, sts))
            if is_ex == 1:
                f_exact.setdefault(a, {})[bd] = szi
                f_exact_ts.setdefault(a, {})[bd] = sts
            else:
                f_comp.setdefault(a, {})[bd] = szi

    spot_by_w: dict[str, tuple[list[int], list[float]]] = {}
    with (RAW / "wallet_spot_ubtc_daily.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("obs_date") == "0000-bootstrap":
                continue
            a = r["address"]
            pair = spot_by_w.get(a)
            if pair is None:
                pair = ([], [])
                spot_by_w[a] = pair
            pair[0].append(int(r["obs_ts_ms"]))
            pair[1].append(float(r["spot_ubtc"]))

    zero_rate_bars = {
        str(c["bar_date"])
        for c in candles
        if c.get("funding_rate_16z") is not None and abs(float(c["funding_rate_16z"])) <= 1e-15
    }
    lb_addrs = {
        r["address"].strip().lower()
        for r in wallets
        if is_ever_on_leaderboard(r) or int(r.get("ever_qualified") or 0) == 1
    }
    models = []
    for w in wallets:
        addr = w["address"].strip().lower()
        if not addr or addr == HLP_VAULT_ADDR or addr not in nav_feat:
            continue
        if lb_addrs and addr not in lb_addrs:
            continue
        raw_exact_map = f_exact.get(addr, {})
        raw_exact_ts = f_exact_ts.get(addr, {})
        raw_comp_map = f_comp.get(addr, {})
        exact_map = (
            {
                bd: szi
                for bd, szi in raw_exact_map.items()
                if max(close_ts_ms_for_16z_bar(bd), int(raw_exact_ts.get(bd, 0) or 0)) <= now_ms
            }
            if now_ms > 0
            else raw_exact_map
        )
        comp_map = (
            {bd: szi for bd, szi in raw_comp_map.items() if _utc_day_bounds_ms(bd)[1] <= now_ms}
            if now_ms > 0
            else raw_comp_map
        )
        btc_max_comp_bar = max(comp_map.keys()) if comp_map else ""
        raw_all_comp = (w.get("all_coins_max_comp_bar") or "").strip()
        causal_all_comp = (
            raw_all_comp
            if (raw_all_comp and (now_ms <= 0 or _utc_day_bounds_ms(raw_all_comp)[1] <= now_ms))
            else ""
        )
        max_comp_bar = max(btc_max_comp_bar, causal_all_comp)
        min_exact_bar = min(exact_map.keys()) if exact_map else ""
        settled_bar = (w.get("settled_16z_bar") or "").strip()
        raw_funding_synced_ms = int(w.get("funding_synced_to_ms") or 0)
        if now_ms > 0 and raw_funding_synced_ms > now_ms:
            closed_bar_now = bar_date_for_16z_open_ms(max(0, now_ms - DAY_MS))
            closed_ms_now = close_ts_ms_for_16z_bar(closed_bar_now)
            if (
                closed_ms_now <= now_ms < (closed_ms_now + SETTLEMENT_INDEXING_GRACE_MS)
                and closed_bar_now not in exact_map
            ):
                funding_synced_to_ms = min(now_ms, closed_ms_now - 1)
            else:
                funding_synced_to_ms = now_ms
        else:
            funding_synced_to_ms = raw_funding_synced_ms
        if settled_bar and close_ts_ms_for_16z_bar(settled_bar) > funding_synced_to_ms:
            settled_bar = ""

        raw_exact_since = (w.get("exact_16z_since_bar") or "").strip()
        if (
            now_ms > 0
            and raw_exact_since
            and close_ts_ms_for_16z_bar(raw_exact_since) > funding_synced_to_ms
        ):
            raw_exact_since = ""
        exact_since = clamp_exact_16z_since_bar(
            raw_exact_since,
            max_comp_bar=max_comp_bar,
            min_exact_bar=min_exact_bar,
            funding_synced_to_ms=funding_synced_to_ms,
        )
        cur_szi = float(w.get("cur_btc_szi") or 0.0)
        pos_synced_at_ms = int(w.get("position_synced_at_ms") or 0)
        complete_fills_until_ms = _resolve_complete_fills_until_ms(
            w,
            fill_pair=fills_by_w.get(addr),
            chain_break_ms=fill_chain_break_ms_by_w.get(addr),
            w_frows=f_rows_by_w.get(addr, []),
            now_ms=now_ms,
        )
        models.append({
            "addr": addr,
            "cur_szi": cur_szi,
            "pos_synced_at_ms": pos_synced_at_ms,
            "complete_fills_until_ms": complete_fills_until_ms,
            "funding_backfilled": bool(int(w.get("funding_backfilled") or 0)),
            "funding_synced_to_ms": funding_synced_to_ms,
            "exact_16z_since_bar": exact_since,
            "settled_16z_bar": settled_bar,
            "f_exact": exact_map,
            "f_comp": comp_map,
            "first_av_ms": nav_feat[addr][0][0],
            "has_prev_obs": False,
            "prev_szi": 0.0,
            "cum_abs_szi": 0.0,
            "cum_diff_szi": 0.0,
            "active_days": 0,
            "flip_days": 0,
        })

    out: dict[str, dict[str, Any]] = {}
    scoped_candles = [
        c for c in candles if c["bar_date"] >= CHURN_WARMUP_START_DATE and int(c["open_ts_ms"]) < now_ms
    ]
    for bi, c in enumerate(scoped_candles):
        bdate = c["bar_date"]
        prev_bdate = scoped_candles[bi - 1]["bar_date"] if bi > 0 else prev_bar_date(bdate)
        close_ms = int(c["close_ts_ms"])
        bar_closed = bool(c["is_complete"]) and now_ms >= close_ms
        t_ms = min(close_ms, now_ms)
        bdate_close_ms = close_ts_ms_for_16z_bar(bdate)
        before_warmup = bdate < MIN_SIGNAL_WARMUP_DATE

        long_btc = short_btc = 0.0
        long_w = short_w = admitted = 0
        for wm in models:
            if t_ms < wm["first_av_ms"]:
                continue
            addr = wm["addr"]
            fe = wm["f_exact"]
            fc = wm["f_comp"]
            if not bar_closed and wm["pos_synced_at_ms"] <= t_ms:
                szi_t = wm["cur_szi"]
            elif wm["complete_fills_until_ms"] > 0 and t_ms <= wm["complete_fills_until_ms"]:
                pair = fills_by_w.get(addr)
                if pair is None or not pair[0]:
                    szi_t = 0.0
                else:
                    ts_f, ep_f = pair
                    idx_f = bisect.bisect_right(ts_f, t_ms) - 1
                    szi_t = ep_f[idx_f] if idx_f >= 0 else 0.0
            elif t_ms >= bdate_close_ms and bdate in fe:
                szi_t = fe[bdate]
            elif (
                t_ms >= bdate_close_ms
                and bdate not in zero_rate_bars
                and bdate_close_ms <= wm["funding_synced_to_ms"]
                and (
                    (wm["exact_16z_since_bar"] and bdate >= wm["exact_16z_since_bar"])
                    or (wm["settled_16z_bar"] and bdate == wm["settled_16z_bar"])
                )
            ):
                szi_t = fe.get(bdate, 0.0)
            elif prev_bdate in fe:
                szi_t = fe[prev_bdate]
            elif wm["funding_backfilled"]:
                bdate_utc_midnight_ms = bdate_close_ms - 16 * HOUR_MS
                comp_bdate = prev_bar_date(prev_bdate) if t_ms < bdate_utc_midnight_ms else prev_bdate
                szi_t = fc.get(comp_bdate, 0.0)
            else:
                szi_t = 0.0

            if wm["has_prev_obs"]:
                wm["cum_diff_szi"] += abs(szi_t - wm["prev_szi"])
                if szi_t * wm["prev_szi"] < -0.01:
                    wm["flip_days"] += 1
            wm["has_prev_obs"] = True
            wm["cum_abs_szi"] += abs(szi_t)
            if abs(szi_t) > 0.05:
                wm["active_days"] += 1
            wm["prev_szi"] = szi_t

            if before_warmup:
                continue
            if (t_ms - wm["first_av_ms"]) / float(DAY_MS) < MIN_HISTORY_DAYS:
                continue
            ts_a, pk_a, pf_a, ns_a = nav_feat[addr]
            idx_a = bisect.bisect_right(ts_a, t_ms) - 1
            if idx_a < 0 or pk_a[idx_a] < MIN_CAPITAL_USD or ns_a[idx_a] < MIN_DIETZ_WEEKS or pf_a[idx_a] < MIN_DIETZ_PF:
                continue
            if wm["active_days"] >= MIN_CHURN_EVAL_DAYS:
                if (wm["cum_diff_szi"] / max(wm["cum_abs_szi"], 1e-6)) > MAX_DAILY_CHURN or (wm["flip_days"] / wm["active_days"]) > MAX_DAILY_FLIP_RATE:
                    continue
            if wm["cum_abs_szi"] > 1e-4:
                admitted += 1
            if abs(szi_t) < 1e-4:
                continue
            sp = spot_by_w.get(addr)
            if szi_t < -0.5 and sp is not None and sp[0] and sp[1]:
                idx_sp = bisect.bisect_right(sp[0], t_ms) - 1
                if idx_sp >= 0 and sp[1][idx_sp] >= HEDGE_MIN_OFFSET_RATIO * abs(szi_t):
                    continue
            if szi_t > 0:
                long_btc += szi_t
                long_w += 1
            else:
                short_btc += abs(szi_t)
                short_w += 1

        if before_warmup:
            continue
        active_w = long_w + short_w
        gross = long_btc + short_btc
        if gross > 1e-6 and active_w >= 5:
            out[bdate] = {
                "l1_wallets": active_w,
                "l1_long_wallets": long_w,
                "l1_short_wallets": short_w,
                "l1_admitted_pool": admitted,
                "l1_long_btc": round(long_btc, 5),
                "l1_short_btc": round(short_btc, 5),
                "l1_net_btc": round(long_btc - short_btc, 5),
                "l1_gross_btc": round(gross, 5),
                "l1_tilt": round((long_btc - short_btc) / gross, 6),
            }
    return out


def average_ranks(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sv = arr[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    if float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return 0.0
    c = float(np.corrcoef(rx, ry)[0, 1])
    return c if math.isfinite(c) else None


def compute_rv20(closes: list[float]) -> list[float | None]:
    n = len(closes)
    rets = [(closes[i] / closes[i - 1] - 1.0) if (i > 0 and closes[i - 1] > 0.0) else 0.0 for i in range(n)]
    out: list[float | None] = [None] * n
    for i in range(20, n):
        seg = np.array(rets[i - 19 : i + 1], dtype=np.float64)
        v = float(np.std(seg, ddof=1)) * math.sqrt(365.0)
        out[i] = v if math.isfinite(v) else None
    return out


def simulate_series(
    comp_bars: list[dict[str, Any]],
    start_date: str,
    *,
    fixed_lev: float | None = None,
    size_mult_by_date: dict[str, float] | None = None,
) -> dict[str, Any]:
    closes = [float(b["btc_close"]) for b in comp_bars]
    rv20 = compute_rv20(closes)
    s_idx = next(i for i, b in enumerate(comp_bars) if b["bar_date"] >= start_date)
    eq = INITIAL_CAPITAL_USD
    peak_eq = eq
    cur_dir = 0
    cur_lev = 0.0
    cur_units = 0.0
    entry_px = stop_px = 0.0
    entry_date = ""
    held_days = 0
    lockout = 0
    dates, dd_pcts, rets = [], [], []
    closed_trades = 0
    won_trades = 0
    trade_eq_start = eq

    for i in range(s_idx, len(comp_bars)):
        b = comp_bars[i]
        bd = b["bar_date"]
        px_o, px_h, px_l, px_c = b["btc_open"], b["btc_high"], b["btc_low"], b["btc_close"]
        prev_c = closes[i - 1] if (i > s_idx and i > 0) else px_c
        tilt = b.get("l1_tilt")
        vol_i = rv20[i]
        eq_prev = eq
        stopped = False

        if cur_dir != 0 and prev_c > 0 and abs(cur_units) > 0:
            held_days += 1
            if cur_dir == 1 and stop_px > 0 and px_l <= stop_px:
                fill_px = min(px_o, stop_px) if px_o < stop_px else stop_px
                eq = max(1.0, eq + cur_units * (fill_px - prev_c) - abs(cur_units) * prev_c * (LONG_FUNDING_BPS_DAY * 1e-4) - abs(cur_units) * fill_px * (FEE_BPS * 1e-4))
                closed_trades += 1
                if eq / trade_eq_start - 1.0 > 0:
                    won_trades += 1
                lockout = 1
                cur_dir, cur_lev, cur_units, entry_px, stop_px, held_days = 0, 0.0, 0.0, 0.0, 0.0, 0
                stopped = True
            elif cur_dir == -1 and stop_px > 0 and px_h >= stop_px:
                fill_px = max(px_o, stop_px) if px_o > stop_px else stop_px
                eq = max(1.0, eq + cur_units * (fill_px - prev_c) - abs(cur_units) * fill_px * (FEE_BPS * 1e-4))
                closed_trades += 1
                if eq / trade_eq_start - 1.0 > 0:
                    won_trades += 1
                lockout = -1
                cur_dir, cur_lev, cur_units, entry_px, stop_px, held_days = 0, 0.0, 0.0, 0.0, 0.0, 0
                stopped = True
            else:
                fund = abs(cur_units) * prev_c * (LONG_FUNDING_BPS_DAY * 1e-4) if cur_dir == 1 else 0.0
                eq = max(1.0, eq + cur_units * (px_c - prev_c) - fund)

        if not stopped:
            day_mult = float(size_mult_by_date.get(bd, 1.0)) if size_mult_by_date else 1.0
            if tilt is not None:
                if lockout == 1 and tilt <= EXIT_TILT:
                    lockout = 0
                elif lockout == -1 and tilt >= -EXIT_TILT:
                    lockout = 0
                next_dir = cur_dir
                if day_mult <= 0.0:
                    next_dir = 0
                elif cur_dir == 0:
                    if tilt >= ENTRY_TILT and lockout != 1:
                        next_dir = 1
                    elif tilt <= -ENTRY_TILT and lockout != -1:
                        next_dir = -1
                elif cur_dir == 1:
                    if tilt <= -ENTRY_TILT:
                        next_dir = -1
                    elif tilt <= EXIT_TILT:
                        next_dir = 0
                elif cur_dir == -1:
                    if tilt >= ENTRY_TILT:
                        next_dir = 1
                    elif tilt >= -EXIT_TILT:
                        next_dir = 0
            else:
                next_dir = 0 if day_mult <= 0.0 else cur_dir

            if next_dir != cur_dir:
                if fixed_lev is not None and next_dir != 0:
                    new_lev = round(float(fixed_lev) * day_mult, 4)
                elif next_dir != 0 and day_mult > 0.0:
                    eff_vol = vol_i if (vol_i is not None and math.isfinite(vol_i) and vol_i > 0.0) else 0.40
                    base_lev = min(MAX_LEV, max(MIN_LEV, VOL_TARGET_NUM / max(eff_vol, 0.20)))
                    new_lev = round(base_lev * day_mult, 4)
                else:
                    new_lev = 0.0
                if cur_dir != 0:
                    eq = max(1.0, eq - abs(cur_units) * px_c * (FEE_BPS * 1e-4))
                    closed_trades += 1
                    if eq / trade_eq_start - 1.0 > 0:
                        won_trades += 1
                cur_dir, cur_lev = next_dir, new_lev
                if cur_dir != 0 and px_c > 0 and new_lev > 0:
                    trade_eq_start = eq
                    open_notional = eq * new_lev
                    eq = max(1.0, eq - open_notional * (FEE_BPS * 1e-4))
                    cur_units = cur_dir * (open_notional / px_c)
                    entry_px = px_c
                    stop_px = px_c * (1.0 - DISASTER_STOP_PCT) if cur_dir == 1 else px_c * (1.0 + DISASTER_STOP_PCT)
                    entry_date = bd
                    held_days = 0
                else:
                    cur_dir, cur_lev, cur_units, entry_px, stop_px, held_days = 0, 0.0, 0.0, 0.0, 0.0, 0

        peak_eq = max(peak_eq, eq)
        dates.append(bd)
        dd_pcts.append(round((eq / peak_eq - 1.0) * 100.0, 2))
        rets.append(eq / eq_prev - 1.0)

    r_arr = np.array(rets, dtype=np.float64)
    sd = float(np.std(r_arr, ddof=1)) if len(r_arr) > 2 else 0.0
    yrs = max(len(dates) / 365.0, 1.0 / 365.0)
    mult = eq / INITIAL_CAPITAL_USD
    cagr_pct = round((math.exp(min(50.0, math.log(mult) / yrs)) - 1.0) * 100.0, 2)
    sharpe = round((float(np.mean(r_arr)) / sd * math.sqrt(365.0)) if sd > 1e-12 else 0.0, 2)
    return {
        "dates": dates,
        "drawdown_pct": dd_pcts,
        "cagr_pct": cagr_pct,
        "max_dd_pct": min(dd_pcts) if dd_pcts else 0.0,
        "sharpe": sharpe,
        "equity_multiple": round(mult, 3),
        "closed_trades": closed_trades,
        "won_trades": won_trades,
        "current_position": {
            "side": "LONG" if cur_dir == 1 else ("SHORT" if cur_dir == -1 else "FLAT"),
            "entry_date": entry_date or None,
            "entry_px": round(entry_px, 2) if entry_px > 0 else None,
            "leverage": round(cur_lev, 4),
            "btc_units": round(cur_units, 5),
        },
    }


def simulate_guarded_backtest(
    candles: list[dict[str, Any]],
    signals: dict[str, dict[str, Any]],
    oi_map: dict[str, float],
    settled_through: str,
) -> dict[str, Any]:
    comp_bars = []
    for c in candles:
        bd = c["bar_date"]
        if c["is_complete"] == 1 and (not settled_through or bd <= settled_through):
            sig = signals.get(bd, {})
            comp_bars.append({
                **c,
                "l1_tilt": sig.get("l1_tilt"),
                "l1_wallets": sig.get("l1_wallets"),
                "l1_gross_btc": sig.get("l1_gross_btc"),
                "hl_btc_oi": oi_map.get(bd),
            })
    quorum_date = next(
        b["bar_date"]
        for b in comp_bars
        if b["bar_date"] >= MIN_SIGNAL_WARMUP_DATE and b.get("l1_tilt") is not None and int(b.get("l1_wallets") or 0) >= MIN_QUORUM_WHALES
    )
    sim_1x = simulate_series(comp_bars, quorum_date, fixed_lev=1.0)
    dd_1x_map = dict(zip(sim_1x["dates"], sim_1x["drawdown_pct"], strict=False))

    q_bars = [b for b in comp_bars if b["bar_date"] >= quorum_date]
    closes = [b["btc_close"] for b in q_bars]
    tilts = [b.get("l1_tilt") for b in q_bars]
    fp_streak = 0
    size_mult_by_date: dict[str, float] = {}
    for t, b in enumerate(q_bars):
        bd = b["bar_date"]
        has_tilt = tilts[t] is not None
        gross_btc = float(b.get("l1_gross_btc") or 0.0) if has_tilt else 0.0
        active_w = int(b.get("l1_wallets") or 0) if has_tilt else 0
        hl_oi = b.get("hl_btc_oi")
        oi_share = (gross_btc / hl_oi) if (hl_oi is not None and hl_oi > 0) else None
        fp_breach = not (
            gross_btc >= FOOTPRINT_MIN_GROSS_BTC
            and (True if oi_share is None else oi_share >= FOOTPRINT_MIN_HL_OI_SHARE)
            and active_w >= FOOTPRINT_MIN_ACTIVE_WHALES
        )
        fp_streak = (fp_streak + 1) if fp_breach else 0

        end_k = t - DECAY_IC_FORWARD_DAYS
        ic5_val = None
        if end_k >= DECAY_IC_WINDOW_DAYS - 1:
            start_k = end_k - (DECAY_IC_WINDOW_DAYS - 1)
            pairs = [
                (tilts[k], closes[k + DECAY_IC_FORWARD_DAYS] / closes[k] - 1.0)
                for k in range(start_k, end_k + 1)
                if tilts[k] is not None and closes[k] > 0 and closes[k + DECAY_IC_FORWARD_DAYS] > 0
            ]
            if len(pairs) >= max(3, DECAY_IC_WINDOW_DAYS // 2):
                ic5_val = spearman_corr(
                    np.array([float(p[0]) for p in pairs], dtype=np.float64),
                    np.array([float(p[1]) for p in pairs], dtype=np.float64),
                )
        dd_1x_pct = float(dd_1x_map.get(bd, 0.0))
        ic_breach = ic5_val is not None and ic5_val < DECAY_MIN_ROLL120_IC
        dd_breach = dd_1x_pct < DECAY_MAX_1X_DD_PCT
        if fp_streak >= FOOTPRINT_STREAK_WARN_DAYS:
            mult = 0.0
        elif (ic_breach and dd_breach) or (ic5_val is not None and ic5_val < DECAY_PAUSE_IC) or (dd_1x_pct < DECAY_PAUSE_1X_DD_PCT):
            mult = 0.0
        elif ic_breach or dd_breach:
            mult = DECAY_CUT_SIZE_MULT
        else:
            mult = 1.0
        size_mult_by_date[bd] = mult

    return simulate_series(comp_bars, quorum_date, size_mult_by_date=size_mult_by_date)


def main() -> None:
    candles, wallets, extra = load_raw_inputs()
    meta = extra["meta"]
    rebuilt_at = meta.get("signals_rebuilt_at") or meta.get("as_of") or "2026-11-22T12:59:23+00:00"
    now_ms = int(round(datetime.fromisoformat(rebuilt_at).timestamp() * 1000))
    reconstructed = reconstruct_signals(candles, wallets, now_ms)

    with (RAW / "l1_daily_signals.csv").open(encoding="utf-8") as f:
        stored_rows = {r["bar_date"]: r for r in csv.DictReader(f)}

    settled_through = meta.get("last_settled_16z_bar") or ""
    settled_dates = [
        c["bar_date"]
        for c in candles
        if c["is_complete"] == 1 and (not settled_through or c["bar_date"] <= settled_through) and c["bar_date"] in stored_rows
    ]
    max_tilt_diff = max(abs(reconstructed[d]["l1_tilt"] - float(stored_rows[d]["ls_tilt"])) for d in settled_dates)
    max_gross_diff = max(abs(reconstructed[d]["l1_gross_btc"] - float(stored_rows[d]["gross_btc"])) for d in settled_dates)
    max_pool_diff = max(abs(reconstructed[d]["l1_admitted_pool"] - int(stored_rows[d]["admitted_pool_size"])) for d in settled_dates)

    bt = simulate_guarded_backtest(candles, reconstructed, extra["oi_map"], settled_through)
    stored_summary = json.loads((BACKTEST / "summary_metrics.json").read_text(encoding="utf-8"))
    stored_metrics = stored_summary["sim_default_from_earliest_quorum"]["metrics"]

    last_settled = settled_dates[-1]
    print(json.dumps({
        "settled_signal_bars": len(settled_dates),
        "max_tilt_diff_vs_stored": round(max_tilt_diff, 8),
        "max_gross_btc_diff_vs_stored": round(max_gross_diff, 8),
        "max_admitted_pool_diff_vs_stored": max_pool_diff,
        "latest_settled_bar": last_settled,
        "latest_settled_signal": reconstructed[last_settled],
        "reproduced_backtest": {
            "cagr_pct": bt["cagr_pct"],
            "stored_cagr_pct": stored_metrics["cagr_pct"],
            "max_dd_pct": bt["max_dd_pct"],
            "sharpe": bt["sharpe"],
            "equity_multiple": bt["equity_multiple"],
            "closed_trades": bt["closed_trades"],
            "current_position": bt["current_position"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
