CONFIG = {
    'n_restaurants': 80,
    'n_consumers': 1000,
    'n_workers': 200,
    'n_periods': 500,
    'strategy_update_interval': 2,
    'platform_strategy': 'HYBRID',  # 可选 'GMV', 'SW', 'HYBRID'
    'HYBRID_LAMBDA': 0.5,
    # 运行多次实验以获取统计
    'DEFAULT_RUNS': 50,
    # 简单的市场范围设置
    'MARKET_GROWTH_RANGE': (0.98, 1.02),
    'COMPETITION_INTENSITY_RANGE': (0.8, 1.2),
    'ECONOMIC_SHOCK_RANGE': (0.90, 1.10),
    'SEASONAL_FACTOR_RANGE': (0.9, 1.1),
    'NETWORK_EFFECT_RANGE': (0.95, 1.05),
    # 佣金/配送费/工资调整
    'COMMISSION_ADJUSTMENT_RATE': 0.03,
    'DELIVERY_FEE_ADJUSTMENT_RATE': 0.03,
    'WAGE_ADJUSTMENT_RATE': 0.03,
    'MIN_COMMISSION': 0.05,
    'MAX_COMMISSION': 0.25,
    'MIN_DELIVERY_FEE': 2.0,
    'MAX_DELIVERY_FEE': 12.0,
    'MIN_WAGE': 3.0,
    'MAX_WAGE': 15.0,
    # 餐厅相关参数
    'MENU_SIZE_RANGE': (4, 8),
    'PRICE_RANGE': (10.0, 20.0),
    'FIXED_COST_RANGE': (100.0, 200.0),
    'QUALITY_RANGE': (0.7, 1.0),
    'REPUTATION_RANGE': (0.5, 0.7),
    'PRICE_MIN': 8.0,
    'PRICE_MAX': 25.0,
    'REVENUE_PRICE_ADJUSTMENT_FACTOR': 0.05,
    'PRICE_GAP_ADJUSTMENT_FACTOR': 0.1,
    # 消费者相关参数
    'CONSUMER_BUDGET_RANGE': (40.0, 100.0),
    'CONSUMER_VALUE_RANGE': (50.0, 120.0),
    'PRICE_SENSITIVITY_RANGE': (0.3, 0.7),
    'TIME_SENSITIVITY_RANGE': (0.2, 0.5),
    'QUALITY_SENSITIVITY_RANGE': (0.4, 0.8),
    'PLATFORM_LOYALTY_RANGE': (0.4, 0.8),
    'REGION_OPTIONS': ['North', 'South', 'East', 'West'],
    'TASTE_PREFERENCE_RANGE': (0.0, 1.0),
    # 配送员相关参数
    'TIME_COST_FACTOR_RANGE': (0.6, 1.0),
    'SKILL_LEVEL_RANGE': (0.7, 1.0),
    'EXPERIENCE_RANGE': (0.1, 0.5),
    'SATISFACTION_RANGE': (0.5, 0.8),
    # 退出机制
    'RESTAURANT_EXIT_THRESHOLD': -300.0,
    'RESTAURANT_TRANSITION_ZONE': -200.0,
    'WORKER_EXIT_THRESHOLD': -50.0,
    'WORKER_TRANSITION_ZONE': -20.0,
    'EMERGENCY_SUBSIDY_RATE': 0.10
}