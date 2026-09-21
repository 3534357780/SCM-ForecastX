"""
服务水平权衡：把「缺货有多值得保」这个判断显式扫一遍。

为什么需要这张表
----------------
目标服务水平由成本比推导：SL = Cu / (Cu + Co)，其中缺货成本 Cu 里含一个
「缺货成本倍数」—— 只算损失的毛利时倍数为 1，计入复购流失与口碑影响时倍数大于 1。
这个倍数不是技术参数，是业务判断。与其替管理层定一个数，不如把整条曲线算出来：

    把缺货看得越贵 -> 目标服务水平越高 -> 安全库存资金越高

管理层或财务挑一个倍数，对应的服务水平与资金占用立刻可得，可复算、可追溯。
持有成本率同理，扫 15% / 20% / 25% 三个常见档位看结论是否翻转。

本模块同时保留蒙特卡洛校验：按正态近似算出的服务水平，在模拟中实际达成多少。
间歇性需求的分布右偏，正态近似系统性偏乐观，这个偏差必须暴露出来而不是藏起来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.risk import monte_carlo_stockout, service_level_from_costs, z_from_service_level


def _ss_value(res: pd.DataFrame, sl: np.ndarray, z: np.ndarray) -> tuple:
    """给定服务水平下的安全库存（件）与资金占用（元）。口径与 plan.py 一致。"""
    L = res["lead_time_days"].to_numpy()
    dm = res["daily_mean"].to_numpy() * res["split_ratio"].to_numpy()
    sd = res["sigma_d_used"].to_numpy() * res["split_ratio"].to_numpy()
    sigma_L = L * C.LEAD_TIME_LN_SIGMA
    ss = z * np.sqrt(L * sd ** 2 + dm ** 2 * sigma_L ** 2)
    return float(ss.sum()), float((ss * res["price"].to_numpy()).sum())


def run() -> pd.DataFrame:
    res = pd.read_csv(C.RISK_FILE)
    rng = np.random.default_rng(C.RANDOM_SEED)
    price = res["price"].to_numpy()
    L = res["lead_time_days"].to_numpy()

    # ---- 主表：缺货成本倍数 -> 目标服务水平 -> 库存资金 ----
    # 复用 risk.service_level_from_costs，扫描表与主口径必须走同一套公式，
    # 否则「敏感性」就在比较两个不同的模型。
    rows = []
    for k in C.STOCKOUT_MULT_SCAN:
        # 扫描时把 ABC 的分档倍数统一换成扫描值，读作「把缺货看得多贵」
        sl = service_level_from_costs(price, None, k_override=k)
        z = z_from_service_level(sl)
        ss_units, ss_value = _ss_value(res, sl, z)
        rows.append({"stockout_multiple": k,
                     "implied_service_level": float(sl.mean()),
                     "z_value": float(np.mean(z)),
                     "total_safety_stock_units": ss_units,
                     "total_safety_stock_value": ss_value})
    out = pd.DataFrame(rows)
    ref = out.loc[out["stockout_multiple"] == 2.0, "total_safety_stock_value"]
    base = float(ref.iloc[0]) if len(ref) else out["total_safety_stock_value"].iloc[0]
    out["vs_mult2_value_pct"] = (out["total_safety_stock_value"] / base - 1) * 100

    # ---- 蒙特卡洛校验：正态近似算出的服务水平，实际达成多少 ----
    achieved = []
    for _, r in out.iterrows():
        z = z_from_service_level(np.array([r["implied_service_level"]]))[0]
        p_sim = []
        for i in range(len(res)):
            dm_i = res["daily_mean"].iloc[i] * res["split_ratio"].iloc[i]
            sd_i = res["sigma_d_used"].iloc[i] * res["split_ratio"].iloc[i]
            if dm_i <= 0:
                continue
            ss_i = z * np.sqrt(L[i] * sd_i ** 2 + dm_i ** 2 * (L[i] * C.LEAD_TIME_LN_SIGMA) ** 2)
            cov = L[i] + ss_i / dm_i
            p, _, _ = monte_carlo_stockout(dm_i, sd_i, L[i], cov,
                                           C.N_SIM_COVERAGE, rng)
            p_sim.append(p)
        achieved.append(1 - float(np.mean(p_sim)))
    out["achieved_service_level_mc"] = achieved
    out["mc_gap_pp"] = (out["achieved_service_level_mc"] - out["implied_service_level"]) * 100

    out.to_csv(C.OUT_DIR / "service_level_tradeoff.csv", index=False)

    # ---- 次表：持有成本率敏感性（验证结论是否依赖该假设）----
    rows2 = []
    for hr in [0.15, 0.20, 0.25]:
        sl = service_level_from_costs(price, None, k_override=2.0, holding_rate=hr)
        z = z_from_service_level(sl)
        ss_units, ss_value = _ss_value(res, sl, z)
        rows2.append({"holding_rate_annual": hr,
                      "implied_service_level": float(sl.mean()),
                      "total_safety_stock_value": ss_value})
    hold = pd.DataFrame(rows2)
    hold["vs_20pct_value_pct"] = (hold["total_safety_stock_value"]
                                  / hold.loc[hold["holding_rate_annual"] == 0.20,
                                             "total_safety_stock_value"].iloc[0] - 1) * 100
    hold.to_csv(C.OUT_DIR / "holding_rate_sensitivity.csv", index=False)

    print("[权衡] 缺货成本倍数 -> 目标服务水平 -> 安全库存资金")
    for _, r in out.iterrows():
        print(f"  倍数 {r.stockout_multiple:>4.1f}x | SL {r.implied_service_level:>6.1%} | "
              f"Z={r.z_value:>5.3f} | 安全库存 {r.total_safety_stock_units:>10,.0f} 件 | "
              f"资金 ${r.total_safety_stock_value:>12,.0f} "
              f"({r.vs_mult2_value_pct:+5.1f}% vs 2x) | "
              f"MC 实际达成 {r.achieved_service_level_mc:>6.1%} "
              f"({r.mc_gap_pp:+.2f} pp)")
    print("\n[权衡] 持有成本率敏感性（缺货倍数固定 2x）")
    for _, r in hold.iterrows():
        print(f"  持有成本率 {r.holding_rate_annual:>5.0%} | "
              f"SL {r.implied_service_level:>6.1%} | "
              f"资金 ${r.total_safety_stock_value:>12,.0f} ({r.vs_20pct_value_pct:+5.1f}%)")
    print(f"[output] {C.OUT_DIR / 'service_level_tradeoff.csv'}")
    print(f"[output] {C.OUT_DIR / 'holding_rate_sensitivity.csv'}")
    return out


if __name__ == "__main__":
    run()
