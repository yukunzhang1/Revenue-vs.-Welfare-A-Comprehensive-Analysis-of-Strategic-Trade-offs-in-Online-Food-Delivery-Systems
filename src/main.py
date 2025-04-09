"""
Main Execution Module for Platform Strategy Simulation
"""

import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict

from simulation.config import CONFIG
from analysis.metrics import (
    run_single_strategy,
    aggregate_results,
    generate_performance_table,
)
from analysis.visualization import (
    plot_strategy_comparison,
    plot_distribution_comparison,
    plot_sensitivity_analysis,
    plot_correlation_matrix,
    plot_scatter_matrix
)
from analysis.sensitivity import run_sensitivity_analysis

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Platform Strategy Simulation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["GMV", "SW", "HYBRID"],
        default=["GMV", "SW", "HYBRID"],
        help="Strategies to compare"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=CONFIG["DEFAULT_RUNS"],
        help="Number of experimental runs per strategy"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=CONFIG["n_periods"],
        help="Simulation periods per run"
    )
    parser.add_argument(
        "--sensitivity-param",
        type=str,
        default="initial_commission",
        help="Parameter for sensitivity analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for figures and tables"
    )

    return parser.parse_args()


def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_arguments()

    # 动态配置实验参数
    base_config = {
        **CONFIG,
        "n_periods": args.periods,
        "platform_strategy": "GMV"  # 将被后续覆盖
    }

    # 运行所有策略实验
    all_strategies_data = {}
    for strategy in args.strategies:
        logging.info(f"Running {strategy} strategy with {args.runs} runs")
        all_strategies_data[strategy] = run_single_strategy(
            config=base_config,
            strategy=strategy,
            n_runs=args.runs
        )

    # 生成可视化结果
    try:
        # 1. 策略对比时序图
        aggregated_data = {
            s: aggregate_results(d)
            for s, d in all_strategies_data.items()
        }
        plot_strategy_comparison(
            data=aggregated_data,
            metrics=["gmv", "sw", "active_restaurants"],
            save_path=args.output_dir
        )

        # 2. 生成性能对比表
        perf_table = generate_performance_table(all_strategies_data)
        perf_table.to_csv(f"{args.output_dir}/performance_table.csv", index=False)
        logging.info(f"\nPerformance Table:\n{perf_table.to_markdown()}")

        # 3. 分布对比图
        final_gmv = {
            s: [run[-1] for run in d["gmv"]]
            for s, d in all_strategies_data.items()
        }
        plot_distribution_comparison(
            data_dict=final_gmv,
            metric_name="Final GMV",
            save_path=args.output_dir
        )

        # 4. 敏感性分析
        sens_df = run_sensitivity_analysis(
            config=base_config,
            param_name=args.sensitivity_param,
            param_values=np.linspace(0.15, 0.25, 5).tolist(),
            strategy="HYBRID",
            n_runs=args.runs // 2  # 减少运行次数以提高速度
        )
        plot_sensitivity_analysis(
            df=sens_df,
            param_name=args.sensitivity_param,
            save_path=args.output_dir
        )

        # 5. 散点矩阵图
        plot_scatter_matrix(
            strategies_data=all_strategies_data,
            save_path=args.output_dir
        )

        # 6. 相关性矩阵
        combined_data = pd.DataFrame({
            f"{s}_{m}": [run[-1] for run in d[m]]
            for s, d in all_strategies_data.items()
            for m in ["gmv", "sw"]
        })
        plot_correlation_matrix(
            data=combined_data,
            save_path=args.output_dir
        )

    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()