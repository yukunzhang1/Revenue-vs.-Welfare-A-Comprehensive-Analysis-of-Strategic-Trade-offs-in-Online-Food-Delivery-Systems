import logging
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from ..simulation.environment import MarketEnvironment


def run_single_strategy(config: dict, strategy: str, n_runs: int) -> Dict[str, List[np.ndarray]]:
    results = defaultdict(list)
    original_strategy = config['platform_strategy']
    config['platform_strategy'] = strategy
    for run in range(n_runs):
        logging.info(f"Running {strategy} experiment {run + 1}/{n_runs}")
        env = MarketEnvironment(config)
        env.initialize()
        for _ in range(config['n_periods']):
            env.simulate_period()
        for metric, series in env.metrics.items():
            results[metric].append(np.array(series))
    config['platform_strategy'] = original_strategy
    return results


def aggregate_results(metrics_results: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    agg = {}
    for metric, runs_list in metrics_results.items():
        arr = np.array(runs_list)  # shape = (n_runs, n_periods)
        mean_series = arr.mean(axis=0)
        agg[metric] = mean_series
    return agg

# 统计分析核心函数
def summary_statistics(data: np.ndarray) -> Dict:
    """计算描述性统计指标（均值、标准差、置信区间等）"""
    if len(data) == 0:
        return {}
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    n = len(data)
    sem = stats.sem(data)
    t_val = stats.t.ppf(0.975, n - 1)
    margin = t_val * sem
    ci_lower = mean_val - margin
    ci_upper = mean_val + margin
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        '25%': q25,
        'median': q50,
        '75%': q75,
        'max': max_val,
        '95% CI': (ci_lower, ci_upper)
    }

def compute_confidence_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float]:
    # 添加缺失的置信区间计算函数
    n = len(data)
    if n <= 1:
        return (np.nan, np.nan) if n == 0 else (data[0], data[0])
    mean = np.mean(data)
    sem = stats.sem(data)
    t = stats.t.ppf((1 + confidence) / 2, n-1)
    margin = t * sem
    return (mean - margin, mean + margin)

def generate_performance_table(strategies_data: Dict) -> pd.DataFrame:
    """生成策略性能对比表格"""
    table_data = []
    for strategy, metrics in strategies_data.items():
        row = {'Strategy': strategy}
        for metric_name, runs in metrics.items():
            # 获取最后一个时间步的数据
            final_values = [run[-1] for run in runs]
            stats = summary_statistics(np.array(final_values))
            row[f'{metric_name}_mean'] = stats['mean']
            row[f'{metric_name}_ci'] = stats['95% CI']
        table_data.append(row)
    return pd.DataFrame(table_data).round(2)


def run_sensitivity_analysis(
        config: dict,
        param_name: str,
        param_values: List[float],
        strategy: str = 'GMV',
        n_runs: int = 3
) -> pd.DataFrame:
    """执行敏感性分析"""
    results = []
    original_val = config[param_name]

    for val in param_values:
        config[param_name] = val
        metrics = run_single_strategy(config, strategy, n_runs)

        # 提取最终结果
        final_gmv = [run[-1] for run in metrics['gmv']]
        final_sw = [run[-1] for run in metrics['sw']]
        # 计算统计量
        gmv_ci = compute_confidence_interval(np.array(final_gmv))
        sw_ci = compute_confidence_interval(np.array(final_sw))

        results.append({
            param_name: val,
            'GMV_mean': np.mean(final_gmv),
            'GMV_ci': gmv_ci,
            'SW_mean': np.mean(final_sw),
            'SW_ci': sw_ci
        })

    config[param_name] = original_val
    return pd.DataFrame(results)
