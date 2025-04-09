from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import List

import numpy as np

from .market_condition import MarketCondition
from ..simulation.config import CONFIG  # 导入配置


class DeliveryWorker:
    id: int
    time_cost_factor: float
    skill_level: float
    experience: float
    satisfaction: float
    is_active: bool = True
    cumulative_orders: int = 0
    cumulative_utility: float = 0
    satisfaction_history: List[float] = field(default_factory=list)
    utility_history: List[float] = field(default_factory=list)
    tracked: bool = False

    @classmethod
    def create_random(cls, wid: int):
        return cls(
            id=wid,
            time_cost_factor=random.uniform(*CONFIG['TIME_COST_FACTOR_RANGE']),
            skill_level=random.uniform(*CONFIG['SKILL_LEVEL_RANGE']),
            experience=random.uniform(*CONFIG['EXPERIENCE_RANGE']),
            satisfaction=random.uniform(*CONFIG['SATISFACTION_RANGE']),
            tracked=False
        )

    def decide_accept_order(self, wage: float, expected_orders: float,
                            delivery_time: float, market_condition: MarketCondition,
                            platform_reputation: float) -> bool:
        effective_wage = wage * market_condition.market_growth
        delivery_time_capped = max(10, min(60, delivery_time))
        time_cost = self.time_cost_factor * delivery_time_capped * (1 - 0.2 * self.experience)
        platform_effect = 0.5 + 0.5 * platform_reputation
        experience_factor = 1 + 0.3 * self.experience
        utility = (effective_wage * expected_orders * experience_factor -
                   time_cost * market_condition.competition_intensity) * platform_effect
        if utility > 0:
            self.satisfaction = min(1.0, self.satisfaction + 0.02)
            return True
        else:
            self.satisfaction = max(0.0, self.satisfaction - 0.02)
            return False

    def calculate_utility(self, wage: float, orders: float,
                          delivery_time: float,
                          market_condition: MarketCondition,
                          platform_reputation: float) -> float:
        effective_wage = wage * market_condition.market_growth
        delivery_time_capped = max(10, min(60, delivery_time))
        time_cost = self.time_cost_factor * delivery_time_capped * (1 - 0.2 * self.experience)
        platform_effect = 0.5 + 0.5 * platform_reputation
        experience_factor = 1 + 0.3 * self.experience
        utility = (effective_wage * orders * experience_factor -
                   time_cost * market_condition.competition_intensity) * platform_effect
        self.cumulative_orders += orders
        self.experience = min(1.0, self.experience + 0.002 * orders)
        self.cumulative_utility += utility
        if orders > 0 and (wage * orders) > 0:
            satisfaction = max(0, min(1, utility / (wage * orders)))
        else:
            satisfaction = 0
        self.satisfaction_history.append(satisfaction)
        if len(self.satisfaction_history) > 10:
            self.satisfaction_history.pop(0)
        if self.tracked:
            self.utility_history.append(self.cumulative_utility)
        return utility
