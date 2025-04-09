from collections import defaultdict

import numpy as np
import pandas as pd

from simulation.config import CONFIG
from analysis.metrics import run_single_strategy, aggregate_results, summary_statistics, run_sensitivity_analysis, \
    generate_performance_table
from analysis.visualization import plot_time_series_comparison, plot_box_distribution, plot_correlation_heatmap, \
    plot_state_evolution, plot_strategy_scatter, plot_histogram_kde


def main():
    strategies = ['GMV', 'SW', 'HYBRID']
    base_config = {
        'n_restaurants': CONFIG['n_restaurants'],
        'n_consumers': CONFIG['n_consumers'],
        'n_workers': CONFIG['n_workers'],
        'n_periods': CONFIG['n_periods'],
        'initial_commission': 0.20,
        'initial_delivery_fee': 8.0,
        'initial_wage': 6.0,
        'strategy_update_interval': CONFIG['strategy_update_interval'],
        'platform_strategy': 'GMV',  # 初始设置，将后续覆盖
    }

    all_strategies_data = {}
    for strat in strategies:
        metrics_data = run_single_strategy(base_config, strat, CONFIG['DEFAULT_RUNS'])
        all_strategies_data[strat] = metrics_data

    # 1. 绘制时间序列图：展示 GMV、SW、平台声誉、活跃餐厅数和活跃配送员数等指标随时间变化的趋势
    if 'GMV' in all_strategies_data and 'SW' in all_strategies_data:
        gmv_agg = aggregate_results(all_strategies_data['GMV'])
        sw_agg = aggregate_results(all_strategies_data['SW'])
        plot_time_series_comparison(gmv_agg, sw_agg)
    else:
        print("GMV 或 SW 策略数据缺失，跳过时序对比图绘制。")

    # 2. 生成并输出最终性能表：显示不同策略下的关键指标（均值及 95% 置信区间等）
    perf_table = generate_performance_table(all_strategies_data)
    print("\n=== Final Performance Table ===")
    print(perf_table)

    # 3. 绘制箱线图：比较不同策略在模拟末期关键指标（GMV、SW）的分布情况
    final_data_box = defaultdict(list)
    for strat, metrics_data in all_strategies_data.items():
        if 'gmv' in metrics_data and 'sw' in metrics_data:
            gmv_vals = [arr[-1] for arr in metrics_data['gmv']]
            sw_vals = [arr[-1] for arr in metrics_data['sw']]
            for v in gmv_vals:
                final_data_box[f'{strat}_GMV'].append(v)
            for v in sw_vals:
                final_data_box[f'{strat}_SW'].append(v)
    plot_box_distribution(final_data_box, "Final GMV/SW Boxplot for each Strategy")

    # 4. 绘制直方图/KDE 分布图：以 GMV 策略的 GMV 为例
    if 'GMV_GMV' in final_data_box:
        plot_histogram_kde(final_data_box['GMV_GMV'], title="GMV Strategy Final GMV Distribution")

    # 5. 进行敏感性分析：对初始佣金 (initial_commission) 进行敏感性分析，绘制误差条图展示不同取值下 GMV 和 SW 的均值及 95% 置信区间
    param_values = [0.15, 0.18, 0.20, 0.23, 0.25]
    sens_df = run_sensitivity_analysis(base_config, 'initial_commission', param_values, strategy='GMV',
                                       n_runs=CONFIG['DEFAULT_RUNS'])
    print("\n=== Sensitivity Analysis (initial_commission) under GMV ===")
    print(sens_df)

    # 6. 绘制散点图：展示各策略下最终 GMV 与 SW 之间的关系
    plot_strategy_scatter(all_strategies_data)

    # 7. 绘制相关性热力图：合并各策略最终的 GMV、SW、活跃餐厅数和活跃配送员数后计算相关性并绘制热力图
    combined_data = {}
    for strat, metrics_data in all_strategies_data.items():
        combined_data[f'{strat}_GMV'] = [arr[-1] for arr in metrics_data['gmv']]
        combined_data[f'{strat}_SW'] = [arr[-1] for arr in metrics_data['sw']]
        combined_data[f'{strat}_Restaurants'] = [arr[-1] for arr in metrics_data['active_restaurants']]
        combined_data[f'{strat}_Workers'] = [arr[-1] for arr in metrics_data['active_workers']]
    plot_correlation_heatmap(combined_data, title="Correlation Heatmap of Final Metrics Across Strategies")

    # 8. 绘制状态变量演化图（动态模型验证）：以 GMV 策略为例，展示平台声誉、活跃餐厅数和活跃配送员数随时间变化的趋势
    gmv_metrics = aggregate_results(all_strategies_data['GMV'])
    plot_state_evolution(gmv_metrics)

    # 9. 输出各策略关键指标的统计摘要（示例：以 GMV 策略的最终 GMV 和 SW 为例）
    gmv_stats = summary_statistics(np.array(all_strategies_data['GMV']['gmv'][-1]))
    sw_stats = summary_statistics(np.array(all_strategies_data['GMV']['sw'][-1]))
    # 注意：将字典包装成列表，以确保 DataFrame 构造正确
    print("\n=== GMV Strategy - Final GMV Statistics ===")
    print(pd.DataFrame([gmv_stats]))
    print("\n=== GMV Strategy - Final SW Statistics ===")
    print(pd.DataFrame([sw_stats]))

if __name__ == "__main__":
    main()