from dataclasses import dataclass, field
import random
from typing import List

import numpy as np

from .market_condition import MarketCondition
from ..simulation.config import CONFIG  # 导入配置


class Consumer:
    id: int
    budget: float
    value: float
    price_sensitivity: float
    time_sensitivity: float
    quality_sensitivity: float
    platform_loyalty: float
    region: str
    taste_preference: float
    satisfaction_history: List[float] = field(default_factory=list)

    @classmethod
    def create_random(cls, cid: int):
        region = random.choice(CONFIG['REGION_OPTIONS'])
        taste_pref = random.uniform(*CONFIG['TASTE_PREFERENCE_RANGE'])
        return cls(
            id=cid,
            budget=random.uniform(*CONFIG['CONSUMER_BUDGET_RANGE']),
            value=random.uniform(*CONFIG['CONSUMER_VALUE_RANGE']),
            price_sensitivity=random.uniform(*CONFIG['PRICE_SENSITIVITY_RANGE']),
            time_sensitivity=random.uniform(*CONFIG['TIME_SENSITIVITY_RANGE']),
            quality_sensitivity=random.uniform(*CONFIG['QUALITY_SENSITIVITY_RANGE']),
            platform_loyalty=random.uniform(*CONFIG['PLATFORM_LOYALTY_RANGE']),
            region=region,
            taste_preference=taste_pref
        )

    def calculate_demand(self, price: float, delivery_fee: float,
                         delivery_time: float, restaurant_quality: float,
                         platform_reputation: float,
                         market_condition: MarketCondition,
                         restaurant_region: str, taste_match: float) -> float:
        effective_price = price + delivery_fee
        if effective_price > self.budget:
            return 0
        distance_factor = 1.2 if restaurant_region != self.region else 1.0
        taste_factor = 1.0 + 0.5 * taste_match
        network_effect = market_condition.network_effect * (0.5 + 0.5 * platform_reputation)
        satisfaction_factor = 1.0
        if self.satisfaction_history:
            satisfaction_factor = 1.0 + 0.2 * (np.mean(self.satisfaction_history) - 0.5)
        quality_effect = restaurant_quality * self.quality_sensitivity
        loyalty_effect = platform_reputation * self.platform_loyalty
        demand = (self.value + quality_effect + loyalty_effect) * network_effect * satisfaction_factor \
                 * taste_factor - self.price_sensitivity * effective_price \
                 - self.time_sensitivity * delivery_time * distance_factor
        return max(0, demand)

    def calculate_utility(self, value: float, price: float,
                          delivery_time: float, quality: float,
                          platform_reputation: float) -> float:
        quality_utility = quality * self.quality_sensitivity
        platform_effect = platform_reputation * self.platform_loyalty
        utility = value + quality_utility + platform_effect - self.time_sensitivity * delivery_time - price
        satisfaction = max(0, min(1, utility / self.value))
        self.satisfaction_history.append(satisfaction)
        if len(self.satisfaction_history) > 10:
            self.satisfaction_history.pop(0)
        return utility
