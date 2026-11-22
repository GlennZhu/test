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

DAY_MS = 86_400_000
HLP_VAULT_ADDR = "0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00"
MIN_CAPITAL_USD = 3_000_000.0
MIN_HISTORY_DAYS = 30.0
MIN_DIETZ_WEEKS = 4
MIN_DIETZ_PF = 1.20
MIN_CHURN_EVAL_DAYS = 20
MAX_DAILY_CHURN = 1.0
MAX_DAILY_FLIP_RATE = 0.20
HEDGE_MIN_OFFSET_RATIO = 0.70

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
    close_dt = datetime.fromtimestamp((open_ts_ms + DAY_MS) / 1000.0, tz=UTC)
    return close_dt.strftime("%Y-%m-%d")


def close_ts_ms_for_16z_bar(bar_date: str) -> int:
    dt = datetime.strptime(bar_date, "%Y-%m-%d").replace(hour=16, minute=0, second=0, tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def build_causal_nav_features(
    raw_av_by_addr: dict[str, list[tuple[int, float, float]]]
) -> dict[str, tuple[list[int], list[float], list[float], list[int]]]:
    min_week_ms = int(6.5 * DAY_MS)
    nav_feat: dict[str, tuple[list[int], list[float], list[float], list[int]]] = {}
    for addr, pts in raw_av_by_addr.items():
        if not pts:
            continue
        ts_list, peak_list, pf_list, weeks_list = [], [], [], []
        peak_av = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        active_weeks = 0
        base_ts, base_av, base_cp = pts[0]
        prev_peak = max(base_av, 0.0)
        for ts_ms, av_val, cum_pnl in pts:
            if av_val > peak_av:
                peak_av = av_val
            if (ts_ms - base_ts) >= min_week_ms:
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
                base_ts, base_av, base_cp = ts_ms, av_val, cum_pnl
                prev_peak = peak_av
            pf = (gross_profit / gross_loss) if gross_loss > 1e-9 else (99.0 if gross_profit > 0.0 else 0.0)
            ts_list.append(ts_ms)
            peak_list.append(peak_av)
            pf_list.append(pf)
            weeks_list.append(active_weeks)
        nav_feat[addr] = (ts_list, peak_list, pf_list, weeks_list)
    return nav_feat


def load_raw_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    meta = json.loads((RAW / "service_meta.json").read_text(encoding="utf-8"))
    with (RAW / "btc_candles_16z.csv").open(encoding="utf-8") as f:
        candles = [
            {
                "bar_date": r["bar_date"],
                "open_ts_ms": int(r["open_ts_ms"]),
                "close_ts_ms": int(r["close_ts_ms"]),
                "btc_open": float(r["open"]),
                "btc_high": float(r["high"]),
                "btc_low": float(r["low"]),
                "btc_close": float(r["close"]),
                "btc_volume": float(r["volume"]),
                "is_complete": int(r["is_complete"]),
            }
            for r in csv.DictReader(f)
        ]
    oi_map = {}
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
    nav_feat = build_causal_nav_features(raw_av)

    fills_by_w: dict[str, tuple[list[int], list[float]]] = {}
    cur_a, ts_b, ep_b = None, [], []
    with (RAW / "wallet_btc_fills.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["address"]
            if a != cur_a:
                if cur_a is not None:
                    fills_by_w[cur_a] = (ts_b, ep_b)
                cur_a, ts_b, ep_b = a, [], []
            ts_b.append(int(r["ts_ms"]))
            ep_b.append(float(r["end_pos"]))
    if cur_a is not None:
        fills_by_w[cur_a] = (ts_b, ep_b)

    f_exact: dict[str, dict[str, float]] = {}
    f_comp: dict[str, dict[str, float]] = {}
    with (RAW / "wallet_daily_funding_szi.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            target = f_exact if int(r["is_exact_16z"]) == 1 else f_comp
            target.setdefault(r["address"], {})[r["bar_date"]] = float(r["szi"])

    spot_by_w: dict[str, tuple[list[int], list[float]]] = {}
    cur_a, sts_b, ssp_b = None, [], []
    with (RAW / "wallet_spot_ubtc_daily.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["address"]
            if a != cur_a:
                if cur_a is not None:
                    spot_by_w[cur_a] = (sts_b, ssp_b)
                cur_a, sts_b, ssp_b = a, [], []
            sts_b.append(int(r["obs_ts_ms"]))
            ssp_b.append(float(r["spot_ubtc"]))
    if cur_a is not None:
        spot_by_w[cur_a] = (sts_b, ssp_b)

    lb_addrs = {r["address"].lower() for r in wallets if int(r["on_leaderboard"]) == 1}
    models = []
    for w in wallets:
        addr = w["address"].strip().lower()
        if not addr or addr == HLP_VAULT_ADDR or addr not in nav_feat:
            continue
        if lb_addrs and addr not in lb_addrs:
            continue
        comp_map = f_comp.get(addr, {})
        max_comp_bar = max(comp_map.keys()) if comp_map else ""
        exact_since = w.get("exact_16z_since_bar") or ""
        if max_comp_bar and exact_since and exact_since <= max_comp_bar:
            exact_since = bar_date_for_16z_open_ms(close_ts_ms_for_16z_bar(max_comp_bar) + DAY_MS)
        models.append({
            "addr": addr,
            "cur_szi": float(w["cur_btc_szi"] or 0.0),
            "is_complete_life": bool(int(w["is_complete_lifetime"])),
            "fills_synced_to_ms": int(w["fills_synced_to_ms"] or 0),
            "funding_backfilled": bool(int(w["funding_backfilled"])),
            "funding_synced_to_ms": int(w["funding_synced_to_ms"] or 0),
            "position_synced_at_ms": int(w["position_synced_at_ms"] or 0),
            "exact_16z_since_bar": exact_since,
            "settled_16z_bar": w.get("settled_16z_bar") or "",
            "first_av_ms": nav_feat[addr][0][0],
            "has_prev_obs": False,
            "prev_szi": 0.0,
            "cum_abs_szi": 0.0,
            "cum_diff_szi": 0.0,
            "active_days": 0,
            "flip_days": 0,
        })

    out: dict[str, dict[str, Any]] = {}
    for bi, c in enumerate(candles):
        bdate = c["bar_date"]
        prev_bdate = candles[bi - 1]["bar_date"] if bi > 0 else bdate
        bar_closed = bool(c["is_complete"])
        t_ms = min(int(c["close_ts_ms"]), now_ms)
        long_btc = short_btc = 0.0
        long_w = short_w = admitted = 0
        for wm in models:
            if t_ms < wm["first_av_ms"]:
                continue
            addr = wm["addr"]
            fe = f_exact.get(addr, {})
            fc = f_comp.get(addr, {})
            if not bar_closed:
                szi_t = wm["cur_szi"]
            elif bdate in fe:
                szi_t = fe[bdate]
            elif wm["is_complete_life"] and addr in fills_by_w and t_ms <= wm["fills_synced_to_ms"]:
                ts_f, ep_f = fills_by_w[addr]
                idx_f = bisect.bisect_right(ts_f, t_ms) - 1
                szi_t = ep_f[idx_f] if idx_f >= 0 else fc.get(prev_bdate, 0.0)
            elif t_ms <= wm["funding_synced_to_ms"] and (
                (wm["exact_16z_since_bar"] and bdate >= wm["exact_16z_since_bar"])
                or (wm["settled_16z_bar"] and bdate == wm["settled_16z_bar"])
            ):
                szi_t = fe.get(bdate, 0.0)
            elif 0 < wm["position_synced_at_ms"] <= t_ms and wm["position_synced_at_ms"] > (t_ms - DAY_MS):
                szi_t = wm["cur_szi"]
            elif prev_bdate in fe:
                szi_t = fe[prev_bdate]
            elif wm["funding_backfilled"]:
                szi_t = fc.get(prev_bdate, 0.0)
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

            if (t_ms - wm["first_av_ms"]) / float(DAY_MS) < MIN_HISTORY_DAYS:
                continue
            ts_a, pk_a, pf_a, ns_a = nav_feat[addr]
            idx_a = bisect.bisect_right(ts_a, t_ms) - 1
            if idx_a < 0 or pk_a[idx_a] < MIN_CAPITAL_USD or ns_a[idx_a] < MIN_DIETZ_WEEKS or pf_a[idx_a] < MIN_DIETZ_PF:
                continue
            if wm["active_days"] >= MIN_CHURN_EVAL_DAYS:
                if (wm["cum_diff_szi"] / max(wm["cum_abs_szi"], 1e-6)) > MAX_DAILY_CHURN or (wm["flip_days"] / wm["active_days"]) > MAX_DAILY_FLIP_RATE:
                    continue
            admitted += 1
            if abs(szi_t) < 1e-4:
                continue
            sp = spot_by_w.get(addr)
            if szi_t < -0.5 and sp is not None:
                idx_sp = bisect.bisect_right(sp[0], t_ms) - 1
                if idx_sp >= 0 and sp[1][idx_sp] >= HEDGE_MIN_OFFSET_RATIO * abs(szi_t):
                    continue
            if szi_t > 0:
                long_btc += szi_t
                long_w += 1
            else:
                short_btc += abs(szi_t)
                short_w += 1

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


def main() -> None:
    candles, wallets, extra = load_raw_inputs()
    as_of = extra["meta"].get("as_of", "2026-11-22T08:00:00+00:00")
    now_ms = int(datetime.fromisoformat(as_of).timestamp() * 1000)
    reconstructed = reconstruct_signals(candles, wallets, now_ms)

    with (RAW / "l1_daily_signals.csv").open(encoding="utf-8") as f:
        stored = {r["bar_date"]: float(r["ls_tilt"]) for r in csv.DictReader(f)}

    settled_dates = [c["bar_date"] for c in candles if c["is_complete"] == 1 and c["bar_date"] in stored]
    max_diff = max(abs(reconstructed[d]["l1_tilt"] - stored[d]) for d in settled_dates if d in reconstructed)
    last_settled = settled_dates[-1]
    print(json.dumps({
        "settled_signal_bars": len(settled_dates),
        "max_tilt_diff_vs_stored": round(max_diff, 8),
        "latest_settled_bar": last_settled,
        "latest_settled_signal": reconstructed[last_settled],
    }, indent=2))


if __name__ == "__main__":
    main()
