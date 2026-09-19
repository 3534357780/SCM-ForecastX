"""
库存与采购计划：安全库存 -> 再订货点 -> 分类分档 -> 行动 SOP。

设计要点
--------
1. **安全库存同时考虑需求波动与交期波动**，不是只算需求：
       SS = Z · sqrt( L·σ_d²  +  d̄²·σ_L² )
   σ_d 来自回测残差（模型实测误差），σ_L 来自供应商交期分布，
   Z 由目标周期服务水平决定（95% -> 1.645）。
   这一条把「需求预测」与「库存决策」打通：
   预测越准 -> σ_d 越小 -> SS 越小 -> 资金占用越少。
2. **不伪造库存。** M5 没有库存字段，因此不假设「当前库存」，而是输出
   「目标库存水位 / 再订货点」，真实落地时对接 WMS 的在手库存做差额。
3. **分类分档驱动差异化策略**：ABC（金额）× XYZ（需求波动）× Kraljic（支出 × 风险）
   三个维度交叉，对应差异化采购策略，而不是一刀切。
4. **双源 SKU 按关系口径算水位**：每条供货关系的安全库存与再订货点，
   基于「该供应商承担的那部分需求」（总需求 × 主备份额）计算 ——
   向主供下单覆盖 75% 的需求，就按 75% 的口径备水位；σ_d 按份额线性
   缩放（主备需求同源的保守近似）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def z_value(sl: float) -> float:
    if sl in C.Z_TABLE:
        return C.Z_TABLE[sl]
    from math import sqrt, erf, erfinv
    return sqrt(2) * erfinv(2 * sl - 1)


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


def run() -> pd.DataFrame:
    res = pd.read_csv(C.RISK_FILE)

    # 需求统计（daily_mean / daily_std / sigma_d）已在 risk 层算好，这里直接复用
    Z = z_value(C.SERVICE_LEVEL)
    L = res["lead_time_days"].to_numpy()
    # 关系级需求 = SKU 总需求 × 该关系份额（双源 SKU 的主备各自按份额口径备水位）
    dm = res["daily_mean"].to_numpy() * res["split_ratio"].to_numpy()
    sd = res["sigma_d"].to_numpy() * res["split_ratio"].to_numpy()   # 主备需求同源的保守近似
    sigma_L = L * C.LEAD_TIME_LN_SIGMA            # 交期标准差（对数正态近似）

    # 联合不确定性下的安全库存
    res["z_value"] = Z
    res["safety_stock"] = Z * np.sqrt(L * sd ** 2 + dm ** 2 * sigma_L ** 2)
    res["reorder_point"] = dm * L + res["safety_stock"]
    res["target_stock"] = res["reorder_point"]    # 目标水位（真实落地需对接 WMS 在手库存）

    # XYZ：按需求变异系数分档（Syntetos 分类，适用于间歇性需求）
    cv = np.where(dm > 0, res["daily_std"] / np.maximum(dm, 1e-6), 0)
    cv2 = cv ** 2
    res["xyz_class"] = np.where(cv2 < 0.49, "X", np.where(cv2 <= 1.0, "Y", "Z"))

    # Kraljic：支出 × 供应风险 两维四象限
    spend_rank = res["annual_spend"].rank(pct=True)
    risk_rank = res["composite_risk"].rank(pct=True)
    res["kraljic"] = np.where((spend_rank >= 0.7) & (risk_rank >= 0.6), "战略",
                       np.where((spend_rank >= 0.7) & (risk_rank < 0.6), "杠杆",
                       np.where((spend_rank < 0.7) & (risk_rank >= 0.6), "瓶颈", "常规")))

    res["action"] = [ACTION_SOP.get((k, r), "常规监控")
                     for k, r in zip(res["kraljic"], res["risk_level"])]

    # 关键 KPI：供应商集中度（HHI，按采购支出份额）；断供概率用 SKU 级联合口径
    share = res.groupby("supplier_id")["annual_spend"].sum() / res["annual_spend"].sum()
    hhi = float((share ** 2).sum())
    top1 = float(share.max())
    sku = res.drop_duplicates("item_id")
    pd.DataFrame([{"hhi": hhi, "top1_share": top1,
                   "avg_safety_stock": float(res["safety_stock"].mean()),
                   "avg_stockout_prob": float(sku["p_stockout_sku"].mean()),
                   "service_level": C.SERVICE_LEVEL}]).to_csv(
        C.OUT_DIR / "scm_kpis.csv", index=False)

    res.to_csv(C.PLAN_FILE, index=False)
    print(f"[plan] {len(res):,} 行 | 平均安全库存 {res['safety_stock'].mean():.2f} 件 | "
          f"Z = {Z}（服务水平 {C.SERVICE_LEVEL:.0%}）")
    print(f"[plan] Kraljic 分布 {res['kraljic'].value_counts().to_dict()}")
    print(f"[plan] 供应商集中度 HHI = {hhi:.4f} | 最大供应商份额 {top1:.1%}")
    print(f"[output] {C.PLAN_FILE}")
    return res


if __name__ == "__main__":
    run()
