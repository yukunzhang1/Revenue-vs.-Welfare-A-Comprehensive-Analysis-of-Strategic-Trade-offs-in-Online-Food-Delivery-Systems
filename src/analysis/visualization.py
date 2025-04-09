"""
Platform Strategy Visualization Module

包含实验结果的可视化函数，支持时间序列对比、敏感性分析、分布可视化等。
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 配置全局绘图样式
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def plot_time_series_comparison(
        gmv_avg: Dict[str, np.ndarray],
        sw_avg: Dict[str, np.ndarray],
        save_path: str = None
) -> None:
    """
    绘制时间序列对比图（GMV vs SW策略）

    参数:
        gmv_avg (dict): GMV策略的聚合指标数据
        sw_avg (dict): SW策略的聚合指标数据
        save_path (str): 图片保存路径（可选）
    """
    metrics = ['gmv', 'sw', 'reputation', 'active_restaurants',
               'active_workers', 'avg_restaurant_utility',
               'avg_consumer_utility', 'avg_worker_utility']

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Time Series Comparison: GMV vs SW Strategy Metrics', fontsize=16)

    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        if metric in gmv_avg and metric in sw_avg:
            ax.plot(gmv_avg[metric], label='GMV Strategy', lw=2, alpha=0.8)
            ax.plot(sw_avg[metric], label='SW Strategy', lw=2, alpha=0.8)
            ax.set_title(metric.upper(), fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/time_series_comparison.png")
    plt.close()


def plot_box_distribution(
        data_dict: Dict[str, List[float]],
        title: str,
        save_path: str = None
) -> None:
    """
    绘制多策略箱线图

    参数:
        data_dict (dict): 形如 {'Strategy1_GMV': [values], ...}
        title (str): 图表标题
        save_path (str): 图片保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(data_dict)

    sns.boxplot(data=df, orient="h", palette="Set2")
    plt.title(title, fontsize=12)
    plt.xlabel("Value", fontsize=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(f"{save_path}/boxplot_{title.lower().replace(' ', '_')}.png")
    plt.close()


def plot_sensitivity_analysis(
        df: pd.DataFrame,
        param_name: str,
        save_path: str = None
) -> None:
    """
    绘制敏感性分析误差条图

    参数:
        df (DataFrame): 敏感性分析结果数据框
        param_name (str): 被分析的参数名称
        save_path (str): 图片保存路径（可选）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # GMV 子图
    ax1.errorbar(df[param_name], df['Final_GMV_Mean'],
                 yerr=[df['Final_GMV_Mean'] - df['Final_GMV_95CI'].apply(lambda x: x[0]),
                       df['Final_GMV_95CI'].apply(lambda x: x[1]) - df['Final_GMV_Mean']],
                 fmt='o-', capsize=5, color='#2ca02c')
    ax1.set_xlabel(param_name, fontsize=10)
    ax1.set_ylabel("Final GMV", fontsize=10)
    ax1.set_title("Sensitivity of GMV to {}".format(param_name), fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # SW 子图
    ax2.errorbar(df[param_name], df['Final_SW_Mean'],
                 yerr=[df['Final_SW_Mean'] - df['Final_SW_95CI'].apply(lambda x: x[0]),
                       df['Final_SW_95CI'].apply(lambda x: x[1]) - df['Final_SW_Mean']],
                 fmt='o-', capsize=5, color='#d62728')
    ax2.set_xlabel(param_name, fontsize=10)
    ax2.set_ylabel("Final SW", fontsize=10)
    ax2.set_title("Sensitivity of SW to {}".format(param_name), fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/sensitivity_{param_name}.png")
    plt.close()


def plot_correlation_heatmap(
        data_dict: Dict[str, List[float]],
        title: str = "Correlation Heatmap",
        save_path: str = None
) -> None:
    """
    绘制指标相关性热力图

    参数:
        data_dict (dict): 形如 {'Metric1': [values], ...}
        title (str): 图表标题
        save_path (str): 图片保存路径（可选）
    """
    df = pd.DataFrame(data_dict)
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                cbar_kws={'label': 'Correlation Coefficient'},
                square=True)
    plt.title(title, fontsize=14)

    if save_path:
        plt.savefig(f"{save_path}/correlation_heatmap.png")
    plt.close()


def plot_state_evolution(
        metrics: Dict[str, np.ndarray],
        save_path: str = None
) -> None:
    """
    绘制平台状态变量演化图

    参数:
        metrics (dict): 包含平台状态指标的数据字典
        save_path (str): 图片保存路径（可选）
    """
    plt.figure(figsize=(10, 6))

    if 'reputation' in metrics:
        plt.plot(metrics['reputation'], label="Reputation", marker='o', markersize=4)
    if 'active_restaurants' in metrics:
        plt.plot(metrics['active_restaurants'], label="Active Restaurants", marker='s', markersize=4)
    if 'active_workers' in metrics:
        plt.plot(metrics['active_workers'], label="Active Workers", marker='^', markersize=4)

    plt.xlabel("Simulation Period", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.title("Evolution of Platform State Variables", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(f"{save_path}/state_evolution.png")
    plt.close()


def plot_strategy_scatter(
        strategies_data: Dict[str, Dict[str, List[np.ndarray]]],
        save_path: str = None
) -> None:
    """
    绘制策略散点图（GMV vs SW）

    参数:
        strategies_data (dict): 各策略的实验结果数据
        save_path (str): 图片保存路径（可选）
    """
    plt.figure(figsize=(8, 6))

    for strategy, data in strategies_data.items():
        if 'gmv' in data and 'sw' in data:
            x = [arr[-1] for arr in data['gmv']]
            y = [arr[-1] for arr in data['sw']]
            plt.scatter(x, y, label=strategy, alpha=0.7, s=80)

    plt.xlabel("Final GMV", fontsize=10)
    plt.ylabel("Final SW", fontsize=10)
    plt.title("Final GMV vs SW by Strategy", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(f"{save_path}/strategy_scatter.png")
    plt.close()


# 可视化专用函数
def plot_histogram_kde(data: List[float], title: str):
    """
    直方图/KDE 分布图：
    分析关键指标（例如 GMV 或 SW）的分布情况，
    判断数据是否呈正态分布或存在偏态，为解释置信区间较大提供依据。
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30, color='#4c72b0')
    plt.title(f"Distribution of {title}", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_strategy_scatter(strategies_results: Dict[str, Dict[str, List[np.ndarray]]]):
    """
    散点图：
    绘制各策略下最终 GMV 与 SW 之间的散点图，
    用于直观展示两者之间的相关性。
    """
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(strategies_results))

    for (strategy, metrics), color in zip(strategies_results.items(), colors):
        final_gmv = [run[-1] for run in metrics['gmv']]
        final_sw = [run[-1] for run in metrics['sw']]

        plt.scatter(final_gmv, final_sw,
                    label=strategy,
                    color=color,
                    s=100,
                    alpha=0.7,
                    edgecolor='w')

    plt.xlabel("Final GMV", fontsize=12)
    plt.ylabel("Final SW", fontsize=12)
    plt.title("GMV vs Social Welfare Trade-off", fontsize=14)
    plt.legend(title="Strategy", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def _save_figure(save_path: str, filename: str):
    if save_path:
        plt.savefig(f"{save_path}/{filename}")
        logging.info(f"Saved plot: {filename}")
