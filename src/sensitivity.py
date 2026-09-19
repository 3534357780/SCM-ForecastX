"""
服务水平敏感性分析：目标服务水平 -> 安全库存 -> 库存资金占用 的权衡曲线。

业务背景
--------
服务水平（cycle service level）不是越高越好：从 95% 提到 99%，
缺货率的改善是线性的（5% -> 1%），但安全库存随 Z 近似线性增长，
且每个百分点对应的库存增量越来越贵。本模块把这条权衡曲线算出来，
回答管理层最常问的问题："把服务水平提到 98%，要多压多少库存资金？"

口径
----
与 plan.py 完全一致：SS = Z · sqrt(L·σ_d² + d̄²·σ_L²)。
σ_d 取回测残差（模型实测误差），σ_L = L · LEAD_TIME_LN_SIGMA。
再通过蒙特卡洛验证该 SS 对应的「实际达成服务水平」，说明正态近似
在本数据上的偏差方向。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.risk import monte_carlo_stockout
from src.plan import z_value


SERVICE_LEVELS = [0.90, 0.95, 0.975, 0.98, 0.99]


def run() -> pd.DataFrame:
    res = pd.read_csv(C.RISK_FILE)
    rng = np.random.default_rng(C.RANDOM_SEED)

    L = res["lead_time_days"].to_numpy()
    split = res["split_ratio"].to_numpy()
    # 与 plan 相同的关系级口径：需求与残差都按主备份额折算
    sd = res["sigma_d"].to_numpy() * split
    dm = res["daily_mean"].to_numpy() * split
    price = res["price"].to_numpy()
    sigma_L = L * C.LEAD_TIME_LN_SIGMA
    base = np.sqrt(L * sd ** 2 + dm ** 2 * sigma_L ** 2)   # Z 前的联合标准差

    rows = []
    for sl in SERVICE_LEVELS:
        Z = z_value(sl)
        ss = Z * base
        ss_value = float(np.sum(ss * price))               # 安全库存资金占用
        rop = dm * L + ss
        rows.append({
            "service_level": sl,
            "z_value": round(Z, 3),
            "total_safety_stock_units": float(ss.sum()),
            "total_safety_stock_value": ss_value,
            "avg_rop": float(rop.mean()),
        })

    out = pd.DataFrame(rows)
    # 相对 95% 基准的变化率
    base_val = out.loc[out["service_level"] == C.SERVICE_LEVEL,
                       "total_safety_stock_value"].iloc[0]
    out["vs_baseline_value_pct"] = (out["total_safety_stock_value"]
                                    / base_val - 1) * 100

    # 蒙特卡洛验证：按 SS 推算的实际达成服务水平（正态近似 vs 模拟）
    verified = []
    for sl in SERVICE_LEVELS:
        Z = z_value(sl)
        p_sim = []
        for i in range(len(res)):
            ss_i = Z * base[i]
            cov = res["lead_time_days"].iloc[i] + (ss_i / dm[i] if dm[i] > 0 else 0)
            p, _, _ = monte_carlo_stockout(dm[i], sd[i],
                                           res["lead_time_days"].iloc[i],
                                           cov, C.N_SIM_COVERAGE, rng)
            p_sim.append(p)
        verified.append(1 - float(np.mean(p_sim)))
    out["achieved_service_level_mc"] = verified

    out.to_csv(C.OUT_DIR / "service_level_tradeoff.csv", index=False)

    print("[tradeoff] 目标服务水平 -> 安全库存资金（相对95%）")
    for _, r in out.iterrows():
        sl_txt = f"{r.service_level:.1%}".replace(".0%", "%")
        print(f"  SL {sl_txt:>4} | Z={r.z_value:>5} | "
              f"SS {r.total_safety_stock_units:>10,.0f} 件 | "
              f"资金 ${r.total_safety_stock_value:>12,.0f} "
              f"({r.vs_baseline_value_pct:+.1f}%) | "
              f"MC 验证达成 {r.achieved_service_level_mc:.1%}")
    print(f"[output] {C.OUT_DIR / 'service_level_tradeoff.csv'}")
    return out


if __name__ == "__main__":
    run()
