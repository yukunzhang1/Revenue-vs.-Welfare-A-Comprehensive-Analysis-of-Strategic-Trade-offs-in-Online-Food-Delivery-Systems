from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import List

import numpy as np
from ..simulation.config import CONFIG  # 导入配置
class Platform:
    def __init__(self, strategy: str, initial_commission: float, initial_delivery_fee: float,
                 initial_wage: float):
        self.strategy = strategy
        self.commission_rate = initial_commission
        self.delivery_fee = initial_delivery_fee
        self.worker_wage = initial_wage
        self.reputation = 0.5
        self.market_share = 0.5
        self.history = defaultdict(list)
        self.bad_trend_count = 0
        self.last_sw = None
        self.last_gmv = None

    def update_strategy(self, gmv, sw,
                        avg_restaurant_utility,
                        avg_worker_utility,
                        avg_consumer_utility,
                        market_condition):
        market_factor = (market_condition.market_growth *
                         market_condition.economic_shock *
                         market_condition.network_effect)
        if self.last_sw is not None and self.last_gmv is not None:
            if sw < self.last_sw or gmv < self.last_gmv:
                self.bad_trend_count += 1
            else:
                self.bad_trend_count = max(0, self.bad_trend_count - 1)
        self.last_sw = sw
        self.last_gmv = gmv
        try:
            if self.strategy == 'GMV':
                if not self.history['gmv'] or gmv <= self.history['gmv'][-1]:
                    self.commission_rate = max(CONFIG['MIN_COMMISSION'],
                                               self.commission_rate * (1 - CONFIG['COMMISSION_ADJUSTMENT_RATE']))
                    self.delivery_fee = max(CONFIG['MIN_DELIVERY_FEE'],
                                            self.delivery_fee * (1 - CONFIG['DELIVERY_FEE_ADJUSTMENT_RATE']))
                else:
                    self.commission_rate = min(CONFIG['MAX_COMMISSION'],
                                               self.commission_rate * (1 + 0.5 * CONFIG['COMMISSION_ADJUSTMENT_RATE']))
            elif self.strategy == 'SW':
                if avg_restaurant_utility < 0:
                    self.commission_rate = max(CONFIG['MIN_COMMISSION'],
                                               self.commission_rate * (1 - 0.5 * CONFIG['COMMISSION_ADJUSTMENT_RATE']))
                if avg_worker_utility < 0:
                    self.worker_wage = min(CONFIG['MAX_WAGE'],
                                           self.worker_wage * (1 + CONFIG['WAGE_ADJUSTMENT_RATE']))
                if (len(self.history['avg_consumer_utility']) > 0 and
                    avg_consumer_utility < self.history['avg_consumer_utility'][-1]):
                    self.delivery_fee = max(CONFIG['MIN_DELIVERY_FEE'],
                                            self.delivery_fee * (1 - 0.5 * CONFIG['DELIVERY_FEE_ADJUSTMENT_RATE']))
            else:  # HYBRID
                lam = CONFIG['HYBRID_LAMBDA']
                combined = lam * gmv + (1 - lam) * sw
                prev_combined = 0
                if len(self.history['gmv']) > 0 and len(self.history['sw']) > 0:
                    prev_combined = lam * self.history['gmv'][-1] + (1 - lam) * self.history['sw'][-1]
                if combined < prev_combined:
                    self.commission_rate = max(CONFIG['MIN_COMMISSION'],
                                               self.commission_rate * (1 - CONFIG['COMMISSION_ADJUSTMENT_RATE']))
                    self.worker_wage = min(CONFIG['MAX_WAGE'],
                                           self.worker_wage * (1 + 0.5 * CONFIG['WAGE_ADJUSTMENT_RATE']))
                else:
                    self.commission_rate = min(CONFIG['MAX_COMMISSION'],
                                               self.commission_rate * (1 + 0.5 * CONFIG['COMMISSION_ADJUSTMENT_RATE']))
            if self.bad_trend_count >= 3:
                self.commission_rate = max(CONFIG['MIN_COMMISSION'],
                                           self.commission_rate * (1 - CONFIG['EMERGENCY_SUBSIDY_RATE']))
                self.worker_wage = min(CONFIG['MAX_WAGE'],
                                       self.worker_wage * (1 + CONFIG['EMERGENCY_SUBSIDY_RATE']))
                self.bad_trend_count = 0
            self.commission_rate *= market_factor
            self.delivery_fee *= market_factor
            self.worker_wage *= market_factor
        except Exception as e:
            logging.error(f"Error in update_strategy: {e}")
        finally:
            self.history['gmv'].append(gmv)
            self.history['sw'].append(sw)
            self.history['avg_consumer_utility'].append(avg_consumer_utility)

    def update_reputation(self, sw, gmv,
                          restaurant_satisfaction,
                          worker_satisfaction,
                          consumer_satisfaction,
                          market_condition):
        try:
            sw_gmv_ratio = sw / (gmv + 1e-6)
            sat_factor = (restaurant_satisfaction + worker_satisfaction + consumer_satisfaction) / 3
            market_effect = (market_condition.market_growth *
                             market_condition.network_effect)
            new_reputation = (0.4 * self.reputation +
                              0.3 * min(1.0, sw_gmv_ratio) +
                              0.3 * sat_factor) * market_effect
            self.reputation = max(0.0, min(1.0, new_reputation))
        except Exception as e:
            logging.error(f"Error in update_reputation: {e}")