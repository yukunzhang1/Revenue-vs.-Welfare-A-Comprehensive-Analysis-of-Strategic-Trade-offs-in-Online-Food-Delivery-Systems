# 定义 MarketCondition, Restaurant, Consumer, DeliveryWorker, Platform 等类
# ----------------- 各类对象定义 -----------------
@dataclass
class MarketCondition:
    market_growth: float
    competition_intensity: float
    economic_shock: float
    seasonal_factor: float
    network_effect: float

    @classmethod
    def generate_random(cls):
        return cls(
            market_growth=random.uniform(*CONFIG['MARKET_GROWTH_RANGE']),
            competition_intensity=random.uniform(*CONFIG['COMPETITION_INTENSITY_RANGE']),
            economic_shock=random.uniform(*CONFIG['ECONOMIC_SHOCK_RANGE']),
            seasonal_factor=random.uniform(*CONFIG['SEASONAL_FACTOR_RANGE']),
            network_effect=random.uniform(*CONFIG['NETWORK_EFFECT_RANGE'])
        )

    def update(self):
        self.competition_intensity *= random.uniform(0.95, 1.05)
        self.economic_shock = random.uniform(*CONFIG['ECONOMIC_SHOCK_RANGE'])
        self.network_effect = min(1.2, self.network_effect * random.uniform(0.98, 1.02))

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

@dataclass
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

@dataclass
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
