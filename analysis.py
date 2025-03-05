# ----------------- 实验结果与统计分析辅助函数 -----------------
def run_single_strategy(config: dict, strategy: str, n_runs: int) -> Dict[str, List[np.ndarray]]:
    results = defaultdict(list)
    original_strategy = config['platform_strategy']
    config['platform_strategy'] = strategy
    for run in range(n_runs):
        logging.info(f"Running {strategy} experiment {run+1}/{n_runs}")
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

def compute_confidence_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float]:
    n = len(data)
    if n <= 1:
        if n == 1:
            return (data[0], data[0])
        return (0, 0)
    mean_val = np.mean(data)
    sem = stats.sem(data)
    t_val = stats.t.ppf((1+confidence)/2, n-1)
    margin = t_val * sem
    return (mean_val - margin, mean_val + margin)

def summary_statistics(data: np.ndarray) -> Dict:
    if len(data) == 0:
        return {}
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    q25, q50, q75 = np.percentile(data, [25,50,75])
    n = len(data)
    sem = stats.sem(data)
    t_val = stats.t.ppf(0.975, n-1)
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
