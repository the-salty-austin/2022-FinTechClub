import datetime
from utility import *
from download import *
from backtest import *

# ==== Backtest Setting ====
SYMBOL = "BTC"  # "ETH", "BNB", "AVAX", "SOL", "DOGE" ...
START  = datetime.datetime(2022, 11, 1, 0, 0)  # (yr,mnth,day,hr,min)
END    = datetime.datetime(2022, 11, 2, 12, 30)

# ==== Strategy Setting ====
GRID_MODE    = 'geometric' # 'arithmetic' or 'geometric'
UPPER_BOUND  = 35000
LOWER_BOUND  = 15000
NUM_OF_GRIDS = 190


df = get_data_since( SYMBOL, START, END )
backtest(df, UPPER_BOUND, LOWER_BOUND, GRID_MODE, NUM=NUM_OF_GRIDS, TX_FEE=0.0005, INVEST=1000, show_tx=False)