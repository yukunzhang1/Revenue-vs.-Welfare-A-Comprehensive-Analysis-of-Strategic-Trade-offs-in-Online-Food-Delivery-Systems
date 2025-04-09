# src/analysis/sensitivity.py

import argparse
import logging
import numpy as np
from typing import List, Dict
from pathlib import Path

import pandas as pd

from ..simulation.config import CONFIG
from metrics import run_single_strategy, compute_confidence_interval
from visualization import plot_sensitivity_analysis


def run_sensitivity_analysis(
        param_name: str,
        param_values: List[float],
        strategy: str = "HYBRID",
        n_runs: int = 30,
        save_path: str = "results/sensitivity"
) -> pd.DataFrame:
    """
    执行参数敏感性分析

    参数:
        param_name (str): 待分析参数名称
        param_values (List[float]): 参数值列表
        strategy (str): 使用的策略类型
        n_runs (int): 每个参数值的运行次数
        save_path (str): 结果保存路径

    返回:
        pd.DataFrame: 包含分析结果的表格
    """
    original_value = CONFIG[param_name]
    results = []

    # 确保输出目录存在
    Path(save_path).mkdir(parents=True, exist_ok=True)

    try:
        for value in param_values:
            # 更新参数值
            CONFIG[param_name] = value
            logging.info(f"Running {n_runs} simulations with {param_name}={value}")

            # 执行实验
            metrics = run_single_strategy(CONFIG, strategy, n_runs)

            # 提取最终结果
            final_gmv = [run[-1] for run in metrics['gmv']]
            final_sw = [run[-1] for run in metrics['sw']]

            # 计算统计量
            gmv_mean = np.mean(final_gmv)
            gmv_ci = compute_confidence_interval(final_gmv)
            sw_mean = np.mean(final_sw)
            sw_ci = compute_confidence_interval(final_sw)

            results.append({
                param_name: value,
                'GMV_mean': gmv_mean,
                'GMV_ci': gmv_ci,
                'SW_mean': sw_mean,
                'SW_ci': sw_ci
            })

        # 生成结果表格
        df = pd.DataFrame(results)
        df.to_csv(f"{save_path}/sensitivity_{param_name}.csv", index=False)

        # 可视化
        plot_sensitivity_analysis(
            df,
            param_name=param_name,
            save_path=save_path
        )

        return df

    finally:
        # 恢复原始参数值
        CONFIG[param_name] = original_value


if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="Run parameter sensitivity analysis")
    parser.add_argument("--param", type=str, required=True,
                        help="Parameter name to analyze")
    parser.add_argument("--values", type=float, nargs="+", required=True,
                        help="List of parameter values")
    parser.add_argument("--strategy", type=str, default="HYBRID",
                        choices=["GMV", "SW", "HYBRID"],
                        help="Strategy type to evaluate")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of runs per parameter value")
    parser.add_argument("--save_path", type=str, default="results/sensitivity",
                        help="Output directory path")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 执行分析
    results_df = run_sensitivity_analysis(
        param_name=args.param,
        param_values=args.values,
        strategy=args.strategy,
        n_runs=args.runs,
        save_path=args.save_path
    )

    # 打印结果摘要
    print("\nSensitivity Analysis Results:")
    print(results_df.round(2))