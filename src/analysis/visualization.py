"""
Platform Strategy Visualization Module

Academic-grade visualization functions for experimental results.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 类型别名定义
MetricsArray = Dict[str, np.ndarray]
StrategyData = Dict[str, Dict[str, List[np.ndarray]]]


# 全局样式配置
def configure_style():
    """配置学术图表样式"""
    plt.style.use('seaborn-paper')
    sns.set_palette("colorblind")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (8, 6)
    })


configure_style()


def _save_figure(save_path: Optional[str], filename: str) -> None:
    """统一保存图表"""
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        full_path = Path(save_path) / filename
        plt.savefig(full_path, dpi=300)
        plt.close()
        logging.info(f"Saved figure: {full_path}")


def plot_strategy_comparison(
        data: Dict[str, MetricsArray],
        metrics: List[str] = ['gmv', 'sw', 'active_restaurants', 'active_workers'],
        save_path: Optional[str] = None
) -> None:
    """
    多指标策略对比时序图

    Parameters:
        data: 包含各策略平均指标数据的字典
        metrics: 需要展示的指标列表
        save_path: 图片保存路径
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    fig.suptitle("Multi-Strategy Metric Comparison", y=1.02)

    for idx, metric in enumerate(metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        for strategy, values in data.items():
            if metric in values:
                ax.plot(values[metric], label=strategy, lw=1.5)

        ax.set_title(metric.upper())
        ax.set_xlabel("Simulation Period")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    _save_figure(save_path, "strategy_comparison.png")


def plot_sensitivity_analysis(
        df: pd.DataFrame,
        param_name: str,
        metrics: List[str] = ['GMV', 'SW'],
        save_path: Optional[str] = None
) -> None:
    """
    增强型敏感性分析图

    Parameters:
        df: 包含敏感性分析结果的数据框
        param_name: 分析的参数名称
        metrics: 需要展示的指标列表
        save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))

    colors = sns.color_palette("husl", len(metrics))
    markers = ['o', 's', '^', 'D']

    for idx, metric in enumerate(metrics):
        mean_col = f'{metric}_mean'
        ci_col = f'{metric}_ci'

        plt.errorbar(
            df[param_name],
            df[mean_col],
            yerr=[df[mean_col] - df[ci_col].str[0],
                  df[ci_col].str[1] - df[mean_col]],
            fmt=markers[idx],
            color=colors[idx],
            label=metric,
            capsize=5,
            markersize=8,
            elinewidth=1.5
        )

    plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title(f"Sensitivity Analysis: {param_name.title()}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(save_path, f"sensitivity_{param_name}.png")


def plot_correlation_matrix(
        data: pd.DataFrame,
        method: str = 'pearson',
        save_path: Optional[str] = None
) -> None:
    """
    增强型相关性矩阵

    Parameters:
        data: 包含指标数据的DataFrame
        method: 相关性计算方法 ('pearson'|'spearman')
        save_path: 图片保存路径
    """
    corr = data.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"{method.title()} Correlation Matrix", fontsize=14)
    _save_figure(save_path, "correlation_matrix.png")


def plot_distribution_comparison(
        data_dict: Dict[str, List[float]],
        metric_name: str,
        save_path: Optional[str] = None
) -> None:
    """
    分布对比图（箱线图+小提琴图）

    Parameters:
        data_dict: 各策略的指标数据字典
        metric_name: 指标名称
        save_path: 图片保存路径
    """
    df = pd.DataFrame([
        {"Strategy": k, "Value": v}
        for k, values in data_dict.items()
        for v in values
    ])

    plt.figure(figsize=(10, 6))

    # 绘制箱线图
    sns.boxplot(
        x="Strategy",
        y="Value",
        data=df,
        width=0.3,
        fliersize=3,
        linewidth=1
    )

    # 叠加小提琴图
    sns.violinplot(
        x="Strategy",
        y="Value",
        data=df,
        inner=None,
        alpha=0.3
    )

    plt.title(f"Distribution Comparison: {metric_name}", fontsize=14)
    plt.xlabel("")
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    _save_figure(save_path, f"distribution_{metric_name}.png")


def plot_scatter_matrix(
        strategies_data: StrategyData,
        metrics: List[str] = ['gmv', 'sw'],
        save_path: Optional[str] = None
) -> None:
    """
    策略散点矩阵图

    Parameters:
        strategies_data: 各策略的实验数据
        metrics: 需要展示的指标列表
        save_path: 图片保存路径
    """
    final_data = []
    for strategy, data in strategies_data.items():
        for metric in metrics:
            if metric in data:
                final_values = [run[-1] for run in data[metric]]
                for val in final_values:
                    final_data.append({
                        'Strategy': strategy,
                        'Metric': metric.upper(),
                        'Value': val
                    })

    df = pd.DataFrame(final_data)
    g = sns.FacetGrid(
        df,
        col="Metric",
        hue="Strategy",
        height=5,
        aspect=1.2,
        sharey=False
    )
    g.map(sns.scatterplot, "Strategy", "Value", alpha=0.7)
    g.add_legend(title="Strategy")

    for ax in g.axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_title(ax.get_title(), fontsize=12)

    _save_figure(save_path, "scatter_matrix.png")