from collections import defaultdict
from typing import List, Dict
import logging
import random
import numpy as np

from ..models.market_condition import MarketCondition
from ..models.restaurant import Restaurant
from ..models.consumer import Consumer
from ..models.worker import DeliveryWorker
from ..models.platform import Platform
from .config import CONFIG

class MarketEnvironment:
    """市场环境，模拟单个策略下的一次实验"""

    def __init__(self, config: dict):
        self.config = config
        self.restaurants: List[Restaurant] = []
        self.consumers: List[Consumer] = []
        self.workers: List[DeliveryWorker] = []
        self.platform: Platform = None
        self.current_period: int = 0
        self.metrics = defaultdict(list)
        self.market_condition = MarketCondition.generate_random()
        self.exit_log = {'restaurant': [], 'worker': []}

    def initialize(self):
        for i in range(self.config['n_restaurants']):
            r = Restaurant.create_random(i, self.current_period)
            if i in [0, 1]:
                r.tracked = True
            self.restaurants.append(r)
        for i in range(self.config['n_consumers']):
            c = Consumer.create_random(i)
            self.consumers.append(c)
        for i in range(self.config['n_workers']):
            w = DeliveryWorker.create_random(i)
            if i in [0, 1]:
                w.tracked = True
            self.workers.append(w)
        self.platform = Platform(
            self.config['platform_strategy'],
            self.config['initial_commission'],
            self.config['initial_delivery_fee'],
            self.config['initial_wage']
        )

    def update_market_condition(self):
        self.market_condition.update()
        # 新餐厅进入
        if random.random() < 0.05 * self.platform.reputation * self.market_condition.market_growth:
            idx = len(self.restaurants)
            r = Restaurant.create_random(idx, self.current_period)
            self.restaurants.append(r)
        # 新配送员进入
        if random.random() < 0.05 * self.platform.reputation * self.market_condition.network_effect:
            idx = len(self.workers)
            w = DeliveryWorker.create_random(idx)
            self.workers.append(w)

    def update_participant_status(self, restaurant_utilities, worker_utilities):
        for r in self.restaurants:
            if r.is_active:
                if r.cumulative_utility < CONFIG['RESTAURANT_EXIT_THRESHOLD']:
                    r.is_active = False
                    self.exit_log['restaurant'].append((self.current_period, r.id, r.cumulative_utility))
                elif r.cumulative_utility < CONFIG['RESTAURANT_TRANSITION_ZONE']:
                    self.platform.commission_rate = max(CONFIG['MIN_COMMISSION'],
                                                        self.platform.commission_rate * (
                                                                    1 - CONFIG['EMERGENCY_SUBSIDY_RATE']))
        for w in self.workers:
            if w.is_active:
                if w.cumulative_utility < CONFIG['WORKER_EXIT_THRESHOLD']:
                    w.is_active = False
                    self.exit_log['worker'].append((self.current_period, w.id, w.cumulative_utility))
                elif w.cumulative_utility < CONFIG['WORKER_TRANSITION_ZONE']:
                    self.platform.worker_wage = min(CONFIG['MAX_WAGE'],
                                                    self.platform.worker_wage * (1 + CONFIG['EMERGENCY_SUBSIDY_RATE']))

    def record_metrics(self, gmv, sw, restaurant_utils, consumer_utils,
                       worker_utils, rest_sat, worker_sat, cons_sat):
        self.metrics['gmv'].append(gmv)
        self.metrics['sw'].append(sw)
        self.metrics['reputation'].append(self.platform.reputation)
        self.metrics['active_restaurants'].append(sum(r.is_active for r in self.restaurants))
        self.metrics['active_workers'].append(sum(w.is_active for w in self.workers))
        self.metrics['avg_restaurant_utility'].append(np.mean(restaurant_utils) if restaurant_utils else 0)
        self.metrics['avg_consumer_utility'].append(np.mean(consumer_utils) if consumer_utils else 0)
        self.metrics['avg_worker_utility'].append(np.mean(worker_utils) if worker_utils else 0)
        self.metrics['restaurant_satisfaction'].append(np.mean(rest_sat) if rest_sat else 0)
        self.metrics['worker_satisfaction'].append(np.mean(worker_sat) if worker_sat else 0)
        # 修正：将“consumer_sat”改为正确的变量“cons_sat”
        self.metrics['consumer_satisfaction'].append(np.mean(cons_sat) if cons_sat else 0)

    def simulate_period(self):
        self.update_market_condition()
        period_gmv = 0
        period_sw = 0
        restaurant_utilities = []
        consumer_utilities = []
        worker_utilities = []
        restaurant_satisfaction = []
        worker_satisfaction = []
        consumer_satisfaction = []
        active_restaurants = [r for r in self.restaurants if r.is_active]
        if active_restaurants:
            avg_market_price = np.mean([np.mean(r.prices) for r in active_restaurants])
        else:
            avg_market_price = 15.0
        # 餐厅更新价格
        for rest in active_restaurants:
            rest.update_prices(self.market_condition, self.platform.reputation, avg_market_price)
        active_workers = [w for w in self.workers if w.is_active]
        if not active_workers or not active_restaurants:
            self.platform.update_reputation(0, 0, 0, 0, 0, self.market_condition)
            self.record_metrics(0, 0, [], [], [], [], [], [])
            self.current_period += 1
            return
        avg_skill = np.mean([w.skill_level for w in active_workers])
        delivery_time = max(10, min(60, 30 * (1 - self.platform.reputation) * (1 - avg_skill)))
        for consumer in self.consumers:
            if not active_restaurants:
                continue
            weights = []
            for rr in active_restaurants:
                taste_match = 1.0 - abs(rr.quality - consumer.taste_preference)
                base_weight = rr.quality * rr.reputation * taste_match
                weights.append(max(0.1, base_weight))
            restaurant = random.choices(active_restaurants, weights=weights)[0]
            menu_item = random.randint(0, restaurant.menu_size - 1)
            price = restaurant.prices[menu_item]
            taste_match2 = 1.0 - abs(restaurant.quality - consumer.taste_preference)
            demand = consumer.calculate_demand(price, self.platform.delivery_fee, delivery_time,
                                               restaurant.quality, self.platform.reputation,
                                               self.market_condition, restaurant.region, taste_match2)
            if demand > 0:
                order_value = price * demand
                period_gmv += order_value
                cons_utility = consumer.calculate_utility(consumer.value * demand,
                                                          order_value + self.platform.delivery_fee,
                                                          delivery_time, restaurant.quality, self.platform.reputation)
                rest_utility = restaurant.calculate_utility(order_value, self.platform.commission_rate,
                                                            self.market_condition, self.platform.reputation)
                consumer_utilities.append(cons_utility)
                restaurant_utilities.append(rest_utility)
                consumer_satisfaction.append(
                    np.mean(consumer.satisfaction_history) if consumer.satisfaction_history else 0.5)
                restaurant_satisfaction.append(max(0, min(1, rest_utility / restaurant.fixed_cost)))
        if period_gmv > 0:
            orders_per_worker = period_gmv / (len(active_workers) * 20.0)
            for wkr in active_workers:
                accept = wkr.decide_accept_order(self.platform.worker_wage, orders_per_worker, delivery_time,
                                                 self.market_condition, self.platform.reputation)
                if accept:
                    w_util = wkr.calculate_utility(self.platform.worker_wage, orders_per_worker, delivery_time,
                                                   self.market_condition, self.platform.reputation)
                    worker_utilities.append(w_util)
                    worker_satisfaction.append(wkr.satisfaction)
        period_sw = sum(consumer_utilities) + sum(restaurant_utilities) + sum(worker_utilities)
        self.platform.update_reputation(period_sw, period_gmv,
                                        np.mean(restaurant_satisfaction) if restaurant_satisfaction else 0,
                                        np.mean(worker_satisfaction) if worker_satisfaction else 0,
                                        np.mean(consumer_satisfaction) if consumer_satisfaction else 0,
                                        self.market_condition)
        if self.current_period % self.config['strategy_update_interval'] == 0:
            self.platform.update_strategy(period_gmv, period_sw,
                                          np.mean(restaurant_utilities) if restaurant_utilities else 0,
                                          np.mean(worker_utilities) if worker_utilities else 0,
                                          np.mean(consumer_utilities) if consumer_utilities else 0,
                                          self.market_condition)
        self.update_participant_status(restaurant_utilities, worker_utilities)
        self.record_metrics(period_gmv, period_sw,
                            restaurant_utilities,
                            consumer_utilities,
                            worker_utilities,
                            restaurant_satisfaction,
                            worker_satisfaction,
                            consumer_satisfaction)  # 这里传入 consumer_satisfaction 参数
        self.current_period += 1