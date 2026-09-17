"""
项目全局配置。

所有可调参数集中在此，每个数字的取值依据见各参数旁注释与 README。
"""
import os
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
# M5 原始文件默认放 data/raw；也可用环境变量指到外部目录，避免大文件进仓库
RAW_DIR   = Path(os.environ.get("M5_RAW_DIR", ROOT / "data" / "raw"))
PROC_DIR  = ROOT / "data" / "processed"    # 清洗后的长表
OUT_DIR   = ROOT / "outputs"               # 结果与图表

# ---------------------------------------------------------------- 数据范围
# 试点范围：单门店 + 单部门。选择依据见 README「范围选择」一节。
STATE = "CA"
STORE = "CA_1"        # 设为 None 则取该州全部门店
DEPT  = "FOODS_3"

# ---------------------------------------------------------------- 回测设定
HORIZON       = 28    # 预测跨度（天），与 M5 官方评测口径一致
N_FOLDS       = 3     # 滚动原点折数，每折 28 天
TRAIN_STEP    = 28    # 训练集内预测原点的采样间隔（天），控制样本量
MIN_HISTORY   = 60    # 最少历史天数，不足不作为训练原点

# ---------------------------------------------------------------- 库存参数
SERVICE_LEVEL = 0.95  # 目标周期服务水平 -> 安全系数 Z
Z_TABLE = {0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.98: 2.054, 0.99: 2.326}

# ---------------------------------------------------------------- 供应商层
# 供应商侧无公开数据，采用「参数化情景模拟」。所有分布参数为专家先验，
# 来源与取值依据见 README「合成层的可辩护性」一节。
RANDOM_SEED   = 42
N_SUPPLIERS   = 120    # 供应商家数（823 个 SKU 由 120 家供应商供货，符合真实格局）
SUPPLIER_SKEW = 0.45   # Dirichlet 集中度参数：越小越集中（少数供应商承接多数 SKU）

N_SIM          = 2000  # 蒙特卡洛模拟次数（主计算）
N_SIM_COVERAGE = 400   # 二分搜索目标服务水平时的模拟次数（降采样提速）

# 准时交付率 OTD ~ Beta(a,b)。Beta(9,1.5) -> 均值 0.857、标准差 0.104，
# 覆盖 0.60~1.00 的真实区间；Beta(18,2) 方差过小会导致所有供应商都是「低风险」。
OTD_ALPHA = 9.0
OTD_BETA  = 1.5
# 质量缺陷率 DPPM ~ 对数正态。中位数 exp(6.3) ≈ 545 DPPM，跨 60~6000 区间。
DPPM_MU    = 6.3
DPPM_SIGMA = 1.1
# 采购价指数 ~ 正态（相对同类中位的偏离）
PRICE_INDEX_SD = 0.13
# 财务健康度 ~ Beta(6, 2.5) -> 均值 0.706
FIN_ALPHA, FIN_BETA = 6.0, 2.5
# 产能弹性 ~ Beta(4, 3) -> 均值 0.571
FLEX_ALPHA, FLEX_BETA = 4.0, 3.0
# 响应天数 ~ Gamma(3, 2.0) -> 均值 6 天
RESP_SHAPE, RESP_SCALE = 3.0, 2.0
# 交期波动（对数正态形状参数，变异系数 ≈ 0.29）
LEAD_TIME_LN_SIGMA = 0.28
# 现状备货口径：交期 + 7 天安全库存（"一周缓冲"，零售快消常见做法）。
# 用于测算「现行政策下的断供概率」，作为改进前的基线。
CURRENT_SAFETY_DAYS = 7.0

REGION_PROFILE = {
    # 区域画像：地缘风险基准分、典型交期倍数
    "当地直供":   {"geo_risk": 8,  "lead_mult": 1.0},
    "近岸供应":   {"geo_risk": 22, "lead_mult": 1.6},
    "远洋供应A":  {"geo_risk": 35, "lead_mult": 3.2},
    "远洋供应B":  {"geo_risk": 48, "lead_mult": 3.8},
}

# QCDSM 五维权重。Delivery 权重最高，因为交付中断对零售缺货的传导最直接；
# Cost 权重低于 Delivery，因为本项目目标是「保供」而非「降本」。
QCDSM_WEIGHTS = {
    "Quality":    0.20,
    "Cost":       0.20,
    "Delivery":   0.30,
    "Service":    0.15,
    "Management": 0.15,
}

# 风险等级阈值（QCDSM 综合分 -> 等级）
GRADE_THRESHOLDS = {"低": 80.0, "中": 65.0}   # >=80 低；65~80 中；<65 高

# ---------------------------------------------------------------- 输出
FORECAST_FILE   = OUT_DIR / "forecast_backtest.csv"
METRICS_FILE    = OUT_DIR / "metrics.csv"
SUPPLIER_FILE   = OUT_DIR / "supplier_master.csv"
RISK_FILE       = OUT_DIR / "supplier_risk.csv"
PLAN_FILE       = OUT_DIR / "replenishment_plan.csv"
NARRATIVE_FILE  = OUT_DIR / "decision_narrative.txt"
