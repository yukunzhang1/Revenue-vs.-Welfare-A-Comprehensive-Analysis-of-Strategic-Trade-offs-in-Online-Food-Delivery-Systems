"""
Market Condition Model

定义市场环境参数及其动态变化规则
"""

from dataclasses import dataclass
import random
from typing import Tuple
from ..simulation.config import CONFIG  # 导入全局配置


@dataclass
class MarketCondition:
    """
    市场条件数据模型，包含动态变化参数

    Attributes:
        market_growth (float): 市场增长率
        competition_intensity (float): 竞争强度
        economic_shock (float): 经济冲击因子
        seasonal_factor (float): 季节性因子
        network_effect (float): 网络效应强度
    """
    market_growth: float
    competition_intensity: float
    economic_shock: float
    seasonal_factor: float
    network_effect: float

    @classmethod
    def generate_random(cls) -> "MarketCondition":
        """生成随机初始市场条件"""
        return cls(
            market_growth=random.uniform(*CONFIG['MARKET_GROWTH_RANGE']),
            competition_intensity=random.uniform(*CONFIG['COMPETITION_INTENSITY_RANGE']),
            economic_shock=random.uniform(*CONFIG['ECONOMIC_SHOCK_RANGE']),
            seasonal_factor=random.uniform(*CONFIG['SEASONAL_FACTOR_RANGE']),
            network_effect=random.uniform(*CONFIG['NETWORK_EFFECT_RANGE'])
        )

    def update(self) -> None:
        """动态更新市场条件"""
        self.competition_intensity *= random.uniform(0.95, 1.05)
        self.economic_shock = random.uniform(*CONFIG['ECONOMIC_SHOCK_RANGE'])
        self.network_effect = min(1.2, self.network_effect * random.uniform(0.98, 1.02))