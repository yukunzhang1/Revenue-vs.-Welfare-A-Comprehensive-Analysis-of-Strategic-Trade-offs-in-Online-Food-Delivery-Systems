import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import random
from collections import defaultdict
import logging
from scipy import stats

#----------------- 设置 Pandas 显示选项，确保完整输出 -----------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ----------------- 随机性控制与日志配置 -----------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
