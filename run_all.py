"""
一键运行全流程。

    python run_all.py

等价于依次执行：
    src.prepare -> src.forecast -> src.risk -> src.plan -> src.narrative
看板单独启动：streamlit run dashboard/app.py
"""
from __future__ import annotations

import time

from src import prepare, forecast, risk, plan, narrative

STEPS = [
    ("数据准备", prepare.build_panel),
    ("需求预测与回测", forecast.run),
    ("供应商风险", risk.run),
    ("库存与采购计划", plan.run),
    ("结构化决策建议", narrative.run),
]


def main() -> None:
    t0 = time.time()
    for name, fn in STEPS:
        print(f"\n{'=' * 62}\n>> {name}\n{'=' * 62}")
        t = time.time()
        fn()
        print(f"-- {name} 完成，耗时 {time.time() - t:.1f}s")
    print(f"\n全流程完成，总耗时 {time.time() - t0:.1f}s")
    print("看板启动：streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
