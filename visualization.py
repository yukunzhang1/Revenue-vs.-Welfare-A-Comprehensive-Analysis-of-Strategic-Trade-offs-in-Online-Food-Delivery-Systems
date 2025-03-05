# ----------------- 可视化函数 -----------------

def plot_time_series_comparison(gmv_avg: Dict[str, np.ndarray], sw_avg: Dict[str, np.ndarray]):
    """
    时间序列图：
    展示在整个模拟周期内，各策略下的指标变化趋势，
    包括 GMV、SW、平台声誉、活跃餐厅数、活跃配送员数及平均效用等，
    便于观察系统是否趋于稳定以及各指标间的动态关系。
    """
    metrics = ['gmv', 'sw', 'reputation', 'active_restaurants', 'active_workers', 
               'avg_restaurant_utility', 'avg_consumer_utility', 'avg_worker_utility']
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Time Series Comparison: GMV vs SW and Other Metrics', fontsize=16)
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        if metric not in gmv_avg or metric not in sw_avg:
            continue
        axes[row, col].plot(gmv_avg[metric], label='GMV Strategy', marker='o', markersize=3)
        axes[row, col].plot(sw_avg[metric], label='SW Strategy', marker='s', markersize=3)
        axes[row, col].set_title(metric)
        axes[row, col].legend()
        axes[row, col].grid(True)
    plt.tight_layout()
    plt.show()

def plot_box_distribution(all_data: Dict[str, List[float]], title: str):
    """
    箱线图：
    比较不同策略在模拟末期关键指标（如 GMV、SW）的分布情况，
    直观展示各策略的中位数、四分位数及异常值。
    """
    df = pd.DataFrame(all_data)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, orient='v')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_histogram_kde(data: List[float], title: str):
    """
    直方图/KDE 分布图：
    分析关键指标（例如 GMV 或 SW）的分布情况，
    判断数据是否呈正态分布或存在偏态，为解释置信区间较大提供依据。
    """
    plt.figure(figsize=(7, 5))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.grid(True)
    plt.show()

def run_sensitivity_analysis(config: dict, param_name: str, param_values: List[float],
                             strategy: str = 'GMV', n_runs: int = 3) -> pd.DataFrame:
    """
    敏感性分析图（误差条图）：
    对关键参数（如 initial_commission）进行敏感性分析，
    展示不同参数取值下 GMV 与 SW 的均值及 95% 置信区间，
    帮助判断该参数对平台整体效益的影响程度。
    """
    results = []
    original_val = config[param_name]
    original_strategy = config['platform_strategy']
    for val in param_values:
        gmv_vals = []
        sw_vals = []
        config[param_name] = val
        config['platform_strategy'] = strategy
        for run in range(n_runs):
            logging.info(f"Sensitivity: {param_name}={val}, run {run+1}/{n_runs}")
            env = MarketEnvironment(config)
            env.initialize()
            for _ in range(config['n_periods']):
                env.simulate_period()
            final_gmv = env.metrics['gmv'][-1] if env.metrics['gmv'] else 0
            final_sw = env.metrics['sw'][-1] if env.metrics['sw'] else 0
            gmv_vals.append(final_gmv)
            sw_vals.append(final_sw)
        gmv_mean = np.mean(gmv_vals)
        sw_mean = np.mean(sw_vals)
        gmv_ci = compute_confidence_interval(np.array(gmv_vals))
        sw_ci = compute_confidence_interval(np.array(sw_vals))
        results.append((val, gmv_mean, gmv_ci, sw_mean, sw_ci))
    config[param_name] = original_val
    config['platform_strategy'] = original_strategy
    df_results = pd.DataFrame(results, columns=[param_name, 'Final_GMV_Mean', 'Final_GMV_95CI', 'Final_SW_Mean', 'Final_SW_95CI'])
    
    # 绘制误差条图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].errorbar(df_results[param_name], df_results['Final_GMV_Mean'],
                     yerr=[df_results['Final_GMV_Mean'] - df_results['Final_GMV_95CI'].apply(lambda x: x[0]),
                           df_results['Final_GMV_95CI'].apply(lambda x: x[1]) - df_results['Final_GMV_Mean']],
                     fmt='-o', capsize=5)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Final GMV")
    axes[0].set_title("Sensitivity Analysis on GMV")
    
    axes[1].errorbar(df_results[param_name], df_results['Final_SW_Mean'],
                     yerr=[df_results['Final_SW_Mean'] - df_results['Final_SW_95CI'].apply(lambda x: x[0]),
                           df_results['Final_SW_95CI'].apply(lambda x: x[1]) - df_results['Final_SW_Mean']],
                     fmt='-o', capsize=5)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("Final SW")
    axes[1].set_title("Sensitivity Analysis on SW")
    
    plt.tight_layout()
    plt.show()
    return df_results

def plot_scatter_gmv_sw(strategies_results: Dict[str, Dict[str, List[np.ndarray]]]):
    """
    散点图：
    绘制各策略下最终 GMV 与 SW 之间的散点图，
    用于直观展示两者之间的相关性。
    """
    plt.figure(figsize=(8,6))
    for strategy, metrics_data in strategies_results.items():
        final_gmv = [arr[-1] for arr in metrics_data['gmv']]
        final_sw = [arr[-1] for arr in metrics_data['sw']]
        plt.scatter(final_gmv, final_sw, label=strategy, alpha=0.7)
    plt.xlabel("Final GMV")
    plt.ylabel("Final SW")
    plt.title("Scatter Plot of Final GMV vs Final SW")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_heatmap(all_data: Dict[str, List[float]], title="Correlation Heatmap"):
    """
    相关性热力图：
    将传入的数据（例如各策略最终的 GMV、SW、活跃餐厅数和活跃配送员数）计算相关性，
    并绘制热力图，帮助探讨各指标之间的关系。
    """
    df = pd.DataFrame(all_data)
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

def plot_state_evolution(agg_metrics: Dict[str, np.ndarray]):
    """
    状态变量演化图：
    绘制平台状态变量（例如平台声誉、活跃餐厅数、活跃配送员数）随时间变化的趋势，
    用于验证动态模型中系统的稳定性与长期行为。
    """
    plt.figure(figsize=(10, 6))
    if 'reputation' in agg_metrics:
        plt.plot(agg_metrics['reputation'], label="Platform Reputation", marker='o')
    if 'active_restaurants' in agg_metrics:
        plt.plot(agg_metrics['active_restaurants'], label="Active Restaurants", marker='s')
    if 'active_workers' in agg_metrics:
        plt.plot(agg_metrics['active_workers'], label="Active Workers", marker='^')
    plt.xlabel("Period")
    plt.ylabel("Value")
    plt.title("Evolution of State Variables Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------- 主函数 -----------------
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
    sens_df = run_sensitivity_analysis(base_config, 'initial_commission', param_values, strategy='GMV', n_runs=CONFIG['DEFAULT_RUNS'])
    print("\n=== Sensitivity Analysis (initial_commission) under GMV ===")
    print(sens_df)
    
    # 6. 绘制散点图：展示各策略下最终 GMV 与 SW 之间的关系
    plot_scatter_gmv_sw(all_strategies_data)
    
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
