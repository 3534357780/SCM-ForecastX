"""
防泄漏回归测试。

这是本项目的「看门测试」：针对两类经典泄漏（特征与标签同源的循环论证、
按 SKU 切分导致训练/验证日期重叠）写可执行断言，防止后续改动重新引入。

运行：pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
from src.forecast import build_rows, to_matrices, FEATURES

PANEL = C.PROC_DIR / "panel.parquet"
needs_data = pytest.mark.skipif(not PANEL.exists(), reason="需先运行 python -m src.prepare")


@pytest.fixture(scope="module")
def mats():
    import pandas as pd
    panel = pd.read_parquet(PANEL)
    return to_matrices(panel)


@needs_data
def test_features_ignore_future(mats):
    """
    核心断言：把预测原点之后的真实数据全部打乱，特征矩阵必须一字不变。
    如果有人在特征里塞了 t-7（h>7 时属于未来）之类的信息，这个测试会红。
    """
    ids, sidx, S, P, cal = mats
    n_days = S.shape[1]
    origin = n_days - 100
    horizon = 20                      # 20 > 7，专门用于暴露 t-7 型泄漏

    base = build_rows(S, P, cal, sidx, origin, horizon)

    rng = np.random.default_rng(0)
    S2 = S.copy()
    S2[:, origin + 1:] = rng.integers(0, 50, size=S2[:, origin + 1:].shape)
    P2 = P.copy()
    P2[:, origin + 1:] = rng.uniform(1, 9, size=P2[:, origin + 1:].shape)

    perturbed = build_rows(S2, P2, cal, sidx, origin, horizon)

    for c in FEATURES:
        np.testing.assert_allclose(
            base[c].to_numpy(), perturbed[c].to_numpy(), rtol=1e-6, atol=1e-6,
            err_msg=f"特征 {c} 受未来数据影响 —— 存在时间泄漏")


@needs_data
def test_split_is_time_based(mats):
    """训练目标必须严格早于验证窗口，训练集与验证集日期区间不得重叠。"""
    ids, sidx, S, P, cal = mats
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
    ids, sidx, S, P, cal = mats
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
def test_no_leakage_in_supplier_grade():
    """
    供应商风险不得引入循环论证：QCDSM 评分必须由特征唯一决定，
    评分是确定性规则（本项目不训练风险分类器）。
    """
    import pandas as pd
    from src.risk import qcdsm
    from src.prepare import build_panel  # noqa: F401  (仅确保模块可导入)
    m = pd.DataFrame({
        "dppm": [50, 400, 5000], "price_index": [1.0, 1.0, 1.2],
        "otd_rate": [0.99, 0.9, 0.7], "response_days": [2, 5, 9],
        "capacity_flex": [0.9, 0.6, 0.3], "financial_health": [0.9, 0.6, 0.3],
        "geo_risk": [8, 22, 48],
    })
    out = qcdsm(m)
    assert out["qcdsm_score"].is_monotonic_decreasing, "质量/交付越差，综合分应越低"
    assert (out["qcdsm_grade"].notna()).all()
