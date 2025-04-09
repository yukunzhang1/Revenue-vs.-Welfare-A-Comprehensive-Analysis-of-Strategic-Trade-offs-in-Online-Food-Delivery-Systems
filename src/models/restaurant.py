from dataclasses import dataclass, field
import random
from typing import List

import numpy as np
from .config import CONFIG  # 导入配置

@dataclass
class Restaurant:
    id: int
    menu_size: int
    prices: List[float]
    fixed_cost: float
    quality: float
    reputation: float
    region: str
    cumulative_revenue: float = 0
    cumulative_utility: float = 0
    is_active: bool = True
    entry_time: int = 0
    utility_history: List[float] = field(default_factory=list)
    tracked: bool = False

    @classmethod
    def create_random(cls, rid: int, current_period: int):
        menu_size = random.randint(*CONFIG['MENU_SIZE_RANGE'])
        region = random.choice(CONFIG['REGION_OPTIONS'])
        return cls(
            id=rid,
            menu_size=menu_size,
            prices=[random.uniform(*CONFIG['PRICE_RANGE']) for _ in range(menu_size)],
            fixed_cost=random.uniform(*CONFIG['FIXED_COST_RANGE']),
            quality=random.uniform(*CONFIG['QUALITY_RANGE']),
            reputation=random.uniform(*CONFIG['REPUTATION_RANGE']),
            region=region,
            entry_time=current_period,
            tracked=False
        )

    def update_prices(self, market_condition: MarketCondition, platform_reputation: float, avg_market_price: float):
        for i in range(self.menu_size):
            market_factor = (market_condition.market_growth *
                             market_condition.economic_shock *
                             market_condition.network_effect)
            competition_factor = 1 - 0.2 * market_condition.competition_intensity
            price_gap = self.prices[i] - avg_market_price
            adjustment = CONFIG['PRICE_GAP_ADJUSTMENT_FACTOR'] * np.tanh(price_gap)
            revenue_factor = np.tanh(self.cumulative_revenue / 10000)
            self.prices[i] *= (1 + CONFIG['REVENUE_PRICE_ADJUSTMENT_FACTOR'] * revenue_factor - adjustment) \
                              * market_factor * competition_factor
            self.prices[i] = max(CONFIG['PRICE_MIN'], min(CONFIG['PRICE_MAX'], self.prices[i]))

    def calculate_utility(self, revenue: float, commission_rate: float,
                          market_condition: MarketCondition, platform_reputation: float) -> float:
        market_impact = (market_condition.market_growth *
                         market_condition.economic_shock *
                         market_condition.network_effect)
        operation_cost = self.fixed_cost * market_condition.competition_intensity
        platform_effect = 0.5 + 0.5 * platform_reputation
        utility = ((1 - commission_rate) * revenue - operation_cost) * market_impact * platform_effect
        self.cumulative_utility += utility
        if self.tracked:
            self.utility_history.append(self.cumulative_utility)
        return utility
