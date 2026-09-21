"""
库存与采购计划：目标服务水平 -> 安全库存 -> 再订货点 -> 分类分档 -> 行动 SOP。

三条口径说明
------------
1. **目标服务水平不是全局拍定的 95%。** 它由「缺一个单位」与「多备一个单位」的
   成本之比推导（见 risk.service_level_from_costs），并按 ABC 自动分档：
   A 类缺货对复购的影响最直接，缺货成本倍数取 3，导出的服务水平最高。
   这一条把「服务水平定多少」从行业惯例变成了可以从财务参数复算的结论。

2. **安全库存用基础波动的σ，不用含促销的σ。**
       SS = Z · sqrt( L·σ_base² + d̄²·σ_L² )
   促销期的波动是一次性的、有促销日历可预见的；基础波动是持续的。混在一起算，
   等于按促销月的剧烈波动给全年备货。促销期的额外需求改由 promo_prebuild_units
   单独预建，账目分开。σ_L 来自供应商交期波动（采购可以谈的那一项）。

3. **不伪造库存。** M5 是 POS 数据，没有在手库存字段，因此不输出「补货量」，
   而是输出「目标库存水位 / 再订货点」，落地时对接 WMS 的在手库存做差额。

分类分档：ABC（金额）× XYZ（需求波动）× Kraljic（支出 × 风险），
三维交叉对应差异化采购策略，而不是一刀切。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# 行动 SOP 映射：Kraljic 象限 × 风险等级 -> 动作
ACTION_SOP = {
    ("战略", "极高"): "双源化 + 提前锁量 + 季度高层复盘",
    ("战略", "高"):   "开发备选源 + 提高安全库存 + 月度交付复盘",
    ("战略", "中"):   "维持双源 + 半年降本谈判",
    ("战略", "低"):   "维持 + 长协锁价",
    ("杠杆", "极高"): "紧急寻源 + 招标比价 + 提高安全库存",
    ("杠杆", "高"):   "引入竞争报价 + 缩短交期条款",
    ("杠杆", "中"):   "年度招标 + 集中采购换价",
    ("杠杆", "低"):   "集中采购 + 自动化下单",
    ("瓶颈", "极高"): "安全库存加倍 + 备用源认证 + 签订保供协议",
    ("瓶颈", "高"):   "提前下单 + 缓冲库存 + 认证替代料",
    ("瓶颈", "中"):   "建立最小库存 + 定期监控交期",
    ("瓶颈", "低"):   "常规监控",
    ("常规", "极高"): "批量下单 + 引入第二供应商",
    ("常规", "高"):   "集中批量下单 + 简化流程",
    ("常规", "中"):   "VMI / 寄售 + 自动补货",
    ("常规", "低"):   "流程自动化，最低管理投入",
}


def _ss(res: pd.DataFrame, sigma_col: str) -> np.ndarray:
    """关系级安全库存：需求与该供应商承担的需求份额同比例缩放。"""
    L = res["lead_time_days"].to_numpy()
    dm = res["daily_mean"].to_numpy() * res["split_ratio"].to_numpy()
    sd = res[sigma_col].to_numpy() * res["split_ratio"].to_numpy()
    sigma_L = L * C.LEAD_TIME_LN_SIGMA
    return np.sqrt(L * sd ** 2 + dm ** 2 * sigma_L ** 2)


def run() -> pd.DataFrame:
    res = pd.read_csv(C.RISK_FILE)

    Z = res["z_value"].to_numpy()                 # 逐关系，由成本比推导
    L = res["lead_time_days"].to_numpy()
    dm = res["daily_mean"].to_numpy() * res["split_ratio"].to_numpy()

    # 主口径：基础波动；对照口径：含促销的全样本波动（旧做法）
    base_new = _ss(res, "sigma_d_used")
    base_old = _ss(res, "sigma_d_all")
    res["safety_stock"] = Z * base_new
    res["reorder_point"] = dm * L + res["safety_stock"]
    res["target_stock"] = res["reorder_point"]
    res["safety_stock_all_sigma"] = Z * base_old
    res["promo_prebuild_units"] = res["promo_prebuild_units"].fillna(0.0)

    # 需求波动分档（XYZ，Syntetos 分类，适用于间歇性需求）
    cv = np.where(res["daily_mean"] > 0,
                  res["daily_std"] / np.maximum(res["daily_mean"], 1e-6), 0)
    res["xyz_class"] = np.where(cv ** 2 < 0.49, "X",
                                np.where(cv ** 2 <= 1.0, "Y", "Z"))

    # Kraljic：支出 × 供应风险
    spend_rank = res["annual_spend"].rank(pct=True)
    risk_rank = res["composite_risk"].rank(pct=True)
    res["kraljic"] = np.where((spend_rank >= 0.7) & (risk_rank >= 0.6), "战略",
                       np.where((spend_rank >= 0.7) & (risk_rank < 0.6), "杠杆",
                       np.where((spend_rank < 0.7) & (risk_rank >= 0.6), "瓶颈", "常规")))
    res["action"] = [ACTION_SOP.get((k, r), "常规监控")
                     for k, r in zip(res["kraljic"], res["risk_level"])]

    # ---- 关键 KPI ----
    price = res["price"].to_numpy()
    ss_new_value = float((res["safety_stock"] * price).sum())
    ss_old_value = float((res["safety_stock_all_sigma"] * price).sum())
    share = res.groupby("supplier_id")["annual_spend"].sum() / res["annual_spend"].sum()
    hhi, top1 = float((share ** 2).sum()), float(share.max())
    sku = res.drop_duplicates("item_id")

    pd.DataFrame([{
        "hhi": hhi, "top1_share": top1,
        "avg_safety_stock": float(res["safety_stock"].mean()),
        "safety_stock_value": ss_new_value,
        "safety_stock_value_all_sigma": ss_old_value,
        "sigma_decomposition_saving_pct": 100 * (1 - ss_new_value / max(ss_old_value, 1e-9)),
        "avg_shortage_prob_current": float(sku["p_shortage_sku"].mean()),
        "avg_service_level_target": float(res["service_level_target"].mean()),
        "promo_prebuild_value": float((res["promo_prebuild_units"] * price).sum()),
        "single_source_sku": int(sku["is_single_source"].sum()),
        "sku_count": int(len(sku)),
    }]).to_csv(C.OUT_DIR / "scm_kpis.csv", index=False)

    # ---- 服务水平分档表（改动③ 的落地结果）----
    # 政策类指标按 SKU 口径（双源 SKU 只算一次），资金类指标按供货关系口径求和。
    sku_lv = res.drop_duplicates("item_id")
    by_class = (sku_lv.groupby("abc_class")
                      .agg(service_level=("service_level_target", "mean"),
                           z_value=("z_value", "mean"),
                           sku=("item_id", "nunique"),
                           target_shortage=("service_level_target",
                                            lambda s: float(1 - s.mean())),
                           shortage_prob_current=("p_shortage", "mean"),
                           coverage_gap_days=("coverage_gap_days", "mean"))
                      .reset_index())
    rel_value = (res.assign(_v=res["safety_stock"] * res["price"])
                    .groupby("abc_class")["_v"].sum().rename("safety_stock_value"))
    by_class = by_class.merge(rel_value, on="abc_class", how="left")
    by_class["safety_stock_value_share"] = (by_class["safety_stock_value"]
                                            / by_class["safety_stock_value"].sum() * 100)
    by_class.to_csv(C.OUT_DIR / "policy_by_class.csv", index=False)

    res.to_csv(C.PLAN_FILE, index=False)

    print(f"[plan] {len(res):,} 行 | 平均安全库存 {res['safety_stock'].mean():.2f} 件 | "
          f"平均目标服务水平 {res['service_level_target'].mean():.2%}"
          f"（逐 SKU 由成本比推导）")
    print(f"[plan] 波动分解：安全库存资金 ${ss_old_value:,.0f} -> ${ss_new_value:,.0f}"
          f"（-{100 * (1 - ss_new_value / max(ss_old_value, 1e-9)):.1f}%）"
          f" 促销预建另计 ${(res['promo_prebuild_units'] * price).sum():,.0f}")
    print(f"[plan] Kraljic 分布 {res['kraljic'].value_counts().to_dict()}")
    print(f"[plan] 供应商集中度 HHI = {hhi:.4f} | 最大供应商份额 {top1:.1%}")
    print(f"[output] {C.PLAN_FILE}")
    print(f"[output] {C.OUT_DIR / 'policy_by_class.csv'}")
    return res


if __name__ == "__main__":
    run()
