# src/analysis/metrics.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from scipy import stats

from src.simulation.environment import MarketEnvironment

# 类型别名定义
MetricsData = Dict[str, List[np.ndarray]]
SimulationConfig = Dict[str, Any]


def run_single_strategy(config: SimulationConfig,
                        strategy: str,
                        n_runs: int) -> MetricsData:
    """
    运行单一策略的多次实验

    Args:
        config: 模拟配置字典
        strategy: 策略名称 ('GMV'/'SW'/'HYBRID')
        n_runs: 实验运行次数

    Returns:
        各指标的多次实验结果（形状为 [n_runs, n_periods]）
    """
    original_strategy = config.get('platform_strategy')
    config['platform_strategy'] = strategy
    results = defaultdict(list)

    try:
        for run_idx in range(n_runs):
            logging.info(f"Running {strategy} experiment {run_idx + 1}/{n_runs}")
            env = MarketEnvironment(config)
            env.initialize()

            # 运行完整周期
            for _ in range(config['n_periods']):
                env.simulate_period()

            # 收集指标
            for metric, series in env.metrics.items():
                results[metric].append(np.array(series))

        return results

    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise
    finally:
        config['platform_strategy'] = original_strategy


def aggregate_results(metrics_results: MetricsData) -> Dict[str, np.ndarray]:
    """
    聚合多次实验结果为平均值

    Args:
        metrics_results: run_single_strategy的输出结果

    Returns:
        各指标的时间序列平均值（形状 [n_periods]）
    """
    return {
        metric: np.array(runs).mean(axis=0)
        for metric, runs in metrics_results.items()
    }


def compute_confidence_interval(data: np.ndarray,
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算置信区间

    Args:
        data: 一维数据数组
        confidence: 置信水平 (默认0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    if len(data) < 2:
        return (np.nan, np.nan)

    mean = np.mean(data)
    sem = stats.sem(data, nan_policy='omit')
    dof = len(data) - 1
    t_value = stats.t.ppf((1 + confidence) / 2, dof)

    margin = t_value * sem
    return (mean - margin, mean + margin)


def summary_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    计算描述性统计量

    Args:
        data: 输入数据数组

    Returns:
        包含统计指标的字典：
        {
            'mean': 均值,
            'std': 标准差,
            'min': 最小值,
            '25%': 第一四分位数,
            'median': 中位数,
            '75%': 第三四分位数,
            'max': 最大值,
            '95% CI': 置信区间
        }
    """
    if len(data) == 0:
        return {}

    return {
        'mean': np.nanmean(data),
        'std': np.nanstd(data),
        'min': np.nanmin(data),
        '25%': np.nanpercentile(data, 25),
        'median': np.nanmedian(data),
        '75%': np.nanpercentile(data, 75),
        'max': np.nanmax(data),
        '95% CI': compute_confidence_interval(data)
    }


def generate_performance_table(strategies_data: Dict[str, MetricsData]) -> pd.DataFrame:
    """
    生成策略对比性能表格

    Args:
        strategies_data: 各策略的实验结果字典

    Returns:
        包含各策略最终指标均值和置信区间的DataFrame
    """
    table_data = []

    for strategy, metrics in strategies_data.items():
        row = {'Strategy': strategy}

        for metric_name, runs in metrics.items():
            try:
                final_values = [run[-1] for run in runs]
                stats = summary_statistics(np.array(final_values))

                # 自动处理单位转换
                if metric_name == 'sw' and stats['mean'] > 1e6:
                    row[f'Final {metric_name.upper()} (M)'] = f"{stats['mean'] / 1e6:.2f} ± {stats['std'] / 1e6:.2f}"
                else:
                    row[f'Final {metric_name.upper()}'] = f"{stats['mean']:.1f} ± {stats['std']:.1f}"

                row[f'{metric_name}_ci'] = stats['95% CI']

            except (IndexError, KeyError) as e:
                logging.warning(f"Missing metric {metric_name} for {strategy}: {e}")
                row[f'Final {metric_name.upper()}'] = 'N/A'

        table_data.append(row)

    # 添加百分比变化列
    df = pd.DataFrame(table_data)
    if 'SW' in df.columns:
        base_value = df[df['Strategy'] == 'SW']['Final SW (M)'].values[0]
        df['SW Change %'] = df['Final SW (M)'].apply(
            lambda x: f"{(float(x.split()[0]) / float(base_value.split()[0]) - 1):.1%}"
        )

    return df.round(2)