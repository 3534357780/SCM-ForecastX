"""
防泄漏回归测试。

这是本项目的「看门测试」：把几类经典泄漏写成可执行断言，防止后续改动重新引入。
  - 特征不得包含预测原点之后的信息（含 t-7 型隐性泄漏）
  - 训练与验证必须按时间切分，不得按 SKU 或行号切分
  - baseline 不得使用未来数据
  - 需求信号的标记与还原（促销日、疑似缺货日）不得使用未来数据
  - 供应商评分是确定性规则，不得出现标签与特征同源的循环论证

运行：pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
from src.forecast import build_rows, to_matrices, FEATURES

PANEL = C.PROC_DIR / "panel.parquet"
needs_data = pytest.mark.skipif(not PANEL.exists(), reason="需先运行 python -m src.prepare")


@pytest.fixture(scope="module")
def mats():
    return to_matrices(pd.read_parquet(PANEL))


@needs_data
def test_features_ignore_future(mats):
    """
    核心断言：把预测原点之后的真实数据全部打乱，特征矩阵必须一字不变。
    如果有人在特征里塞了 t-7（h>7 时属于未来）之类的信息，这个测试会红。
    """
    ids, sidx, S, Sr, M, P, cal = mats
    origin = S.shape[1] - 100
    horizon = 20                      # 20 > 7，专门用于暴露 t-7 型泄漏

    base = build_rows(S, P, cal, sidx, origin, horizon, Sr=Sr, M=M)

    rng = np.random.default_rng(0)
    S2, P2 = S.copy(), P.copy()
    S2[:, origin + 1:] = rng.integers(0, 50, size=S2[:, origin + 1:].shape)
    P2[:, origin + 1:] = rng.uniform(1, 9, size=P2[:, origin + 1:].shape)

    perturbed = build_rows(S2, P2, cal, sidx, origin, horizon, Sr=Sr, M=M)

    for c in FEATURES:
        np.testing.assert_allclose(
            base[c].to_numpy(), perturbed[c].to_numpy(), rtol=1e-6, atol=1e-6,
            err_msg=f"特征 {c} 受未来数据影响 —— 存在时间泄漏")


@needs_data
def test_split_is_time_based(mats):
    """训练目标必须严格早于验证窗口，训练集与验证集日期区间不得重叠。"""
    ids, sidx, S, Sr, M, P, cal = mats
    n_days = S.shape[1]
    H, F = C.HORIZON, C.N_FOLDS
    val_starts = [n_days - k * H - H for k in range(F)][::-1]
    earliest_val = min(val_starts)
    train_origins = list(range(C.MIN_HISTORY, earliest_val - H, C.TRAIN_STEP))
    last_train_target = train_origins[-1] + H
    assert last_train_target < earliest_val, (
        f"训练最晚目标 t={last_train_target} 未严格早于验证起点 t={earliest_val}")


@needs_data
def test_baselines_use_only_past(mats):
    """baseline 的预测也必须只用原点及以前的信息。"""
    from src.forecast import baselines
    ids, sidx, S, Sr, M, P, cal = mats
    origin = S.shape[1] - 100
    rng = np.random.default_rng(1)
    S2 = S.copy()
    S2[:, origin + 1:] = rng.integers(0, 50, size=S2[:, origin + 1:].shape)
    for h in (1, 7, 8, 15, 28):
        b1 = baselines(S, origin, h)
        b2 = baselines(S2, origin, h)
        for k in b1:
            np.testing.assert_allclose(b1[k], b2[k], err_msg=f"{k} 在 h={h} 使用了未来数据")


@needs_data
def test_signal_flags_ignore_future():
    """
    促销标记与缺货还原同样不得使用未来数据。

    这两步产出的 sales_restored 会被当作训练目标，一旦掺入未来信息，
    就等价于把未来销量写进训练标签 —— 比特征泄漏更隐蔽。这里用同样的
    「打乱未来」方法断言历史部分的标记与还原值不变。
    """
    from src.prepare import mark_promo, detect_and_restore

    panel = pd.read_parquet(PANEL)
    origin = int(panel["t"].max()) - 60

    a = detect_and_restore(mark_promo(panel.copy()))

    rng = np.random.default_rng(7)
    p2 = panel.copy()
    fut = (p2["t"] > origin).to_numpy()
    for col, vals in [("sales", rng.integers(0, 60, size=int(fut.sum()))),
                      ("price", rng.uniform(1, 9, size=int(fut.sum())))]:
        p2[col] = p2[col].astype(float)
        p2.loc[fut, col] = vals
    b = detect_and_restore(mark_promo(p2))

    hist = ~fut
    for col in ["is_promo", "is_stockout", "sales_restored"]:
        np.testing.assert_allclose(
            a.loc[hist, col].to_numpy(float), b.loc[hist, col].to_numpy(float),
            rtol=1e-6, atol=1e-6,
            err_msg=f"{col} 受未来数据影响 —— 需求信号处理存在时间泄漏")


@needs_data
def test_scorecard_has_no_circular_label():
    """
    供应商评分不得引入循环论证：记分卡是确定性规则（本项目不训练风险分类器），
    分数必须由特征唯一决定，且随各维度恶化而单调下降。
    """
    from src.risk import scorecard

    m = pd.DataFrame({
        "supplier_id": ["S1", "S2", "S3"],
        "region": ["当地直供", "近岸供应", "远洋供应B"],
        "lead_time_days": [10.0, 16.0, 40.0],
        "lead_time_cv": [0.28, 0.28, 0.28],
        "otd_rate": [0.99, 0.92, 0.75],
        "otif_rate": [0.91, 0.85, 0.69],
        "dppm": [50, 400, 5000],
        "capacity_flex": [0.9, 0.6, 0.3],
        "response_days": [2.0, 5.0, 9.0],
        "financial_health": [0.9, 0.6, 0.3],
        "ttr_days": [5.0, 20.0, 60.0],
        "price_index": [1.0, 1.0, 1.2],
        "geo_risk": [8, 22, 48],
        "region_disrupt_p": [0.03, 0.05, 0.10],
    })
    out = scorecard(m)
    assert out["supplier_score"].is_monotonic_decreasing, "各维度越差，综合分应越低"
    assert (out["tier"].notna()).all()
    # 分数只由特征决定：同样的输入必须得到同样的输出
    again = scorecard(m)
    np.testing.assert_allclose(out["supplier_score"].to_numpy(),
                               again["supplier_score"].to_numpy())


def test_service_level_from_cost_is_monotone():
    """
    目标服务水平必须随「缺货成本倍数」单调上升，且落在合理区间内。
    这是改动③可辩护性的最小验证：结论方向不能反。

    同时断言敏感性扫描与主口径走的是同一套公式 —— 两者不一致，
    「敏感性分析」就变成了在比较两个不同的模型。
    """
    from src.risk import service_level_from_costs

    price = np.array([3.0, 3.0, 3.0, 3.0])
    lv = [service_level_from_costs(price, np.array(["C"] * 4))[0]]
    for abc in ["B", "A"]:
        lv.append(service_level_from_costs(price, np.array([abc] * 4))[0])
    assert lv == sorted(lv), f"服务水平应随缺货成本上升：{lv}"
    assert all(0.5 <= v <= 0.999 for v in lv), f"服务水平越界：{lv}"

    # 扫描口径必须与 ABC 分档口径同源：A 类的分档倍数就是 3x
    a_class = service_level_from_costs(price, np.array(["A"] * 4))[0]
    scanned = service_level_from_costs(price, None, k_override=3.0)[0]
    assert abs(a_class - scanned) < 1e-12, "扫描口径与主口径不一致"
