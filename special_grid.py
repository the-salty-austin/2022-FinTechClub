import pandas as pd
import numpy  as np
import math
import datetime

from utility import timer

class BadMarketBuyError(Exception):
    """Exception raised for market orders at prices higher than limit buy.

    Attributes:
        current_price -- input salary which caused the error\n
        message -- explanation of the error
    """

    def __init__(self, limit, current_price):
        self.message = f"MKT={current_price} > LIM={limit}\nMARKET ORDER should be excuted at prices lower than its LIMIT."
        super().__init__(self.message)

class BadMarketSellError(Exception):
    """Exception raised for market orders at prices lower than limit sell.

    Attributes:
        current_price -- input salary which caused the error\n
        message -- explanation of the error
    """

    def __init__(self, limit, current_price):
        self.message = f"MKT={current_price} < LIM={limit}\nMARKET ORDER should be excuted at prices higher than its LIMIT."
        super().__init__(self.message)

class WrongOrderTypeError(Exception):
    """Exception raised for non existent order types.

    Attributes:
        current_price -- input salary which caused the error\n
        message -- explanation of the error
    """

    def __init__(self, mode):
        self.message = f"\"{mode}\" is an undefined order type."
        super().__init__(self.message)

class BadGridBoundsError(Exception):
    """Exception raised for non existent order types.

    Attributes:
        current_price -- input salary which caused the error\n
        message -- explanation of the error
    """

    def __init__(self, lower, upper):
        self.message = f"LOWER={lower} >= UPPER={upper}. Unreasonable grid bounds."
        super().__init__(self.message)


class Account:
    def __init__(self, initial_cash: float) -> None:
        self.initial_balance = initial_cash
        self.balance         = initial_cash
        self.gridprofit      = 0
        self.next_buy_index  = -2
        self.next_sell_index = -2
        self.records = []
        self.retain_buy  = []
        self.retain_sell = []

class Grid():
    def __init__(self, index: int, limitbuy: float, limitsell: float) -> None:
        self.index     = index
        self.limitbuy  = limitbuy
        self.limitsell = limitsell
        self.quantity  = 0.0
        self.buyprice  = 0.0

    # def buy(self, current_price: float, cash: float, mode: str) -> None:
    def buy(self, current_price: float, quantity: float, mode: str, txfee: float) -> None:
        '''mode: [LIM, MKT]'''
        # limit buy or market buy
        # market buy must be lower than limit buy
        price = 0
        if mode=='LIM':
            price = self.limitbuy
        elif mode=='MKT':
            if current_price <= self.limitbuy:
                price = current_price
            else:
                raise BadMarketBuyError(self.limitbuy, current_price)
        else:
            raise WrongOrderTypeError(mode)

        if current_price <= self.limitbuy:
            # self.quantity = cash * (1-TXFEE) / price
            self.quantity = quantity
            self.buyprice = price

            used_cash = self.quantity * self.buyprice / (1-txfee)
            return used_cash
        return 0

    def sell(self, current_price: float, mode: str, txfee: float) -> float:
        price = 0
        if mode=='LIM':
            price = self.limitsell
        elif mode=='MKT':
            if self.limitsell <= current_price:
                price = current_price
            else:
                raise BadMarketSellError(self.limitbuy, current_price)
        else:
            raise WrongOrderTypeError(mode)

        if current_price >= self.limitsell:
            profit = (price * (1-txfee) - self.buyprice / (1-txfee)) * self.quantity
            returncash = price * self.quantity * (1-txfee)
            self.quantity = 0.0
            self.buyprice = 0.0
            return returncash, profit
        return 0, 0


# https://stackoverflow.com/questions/63020750/how-to-find-average-directional-movement-for-stocks-using-pandas

def ADX(data: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Computes the ADX indicator.
    """
    
    _df = data.copy()
    alpha = 1/period

    # TR
    _df['H-L'] = _df['high'] - _df['low']
    _df['H-C'] = np.abs(_df['high'] - _df['close'].shift(1))
    _df['L-C'] = np.abs(_df['low'] - _df['close'].shift(1))
    _df['TR'] = _df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del _df['H-L'], _df['H-C'], _df['L-C']

    # ATR
    _df['ATR'] = _df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    _df['H-pH'] = _df['high'] - _df['high'].shift(1)
    _df['pL-L'] = _df['low'].shift(1) - _df['low']
    _df['+DX'] = np.where(
        (_df['H-pH'] > _df['pL-L']) & (_df['H-pH']>0),
        _df['H-pH'],
        0.0
    )
    _df['-DX'] = np.where(
        (_df['H-pH'] < _df['pL-L']) & (_df['pL-L']>0),
        _df['pL-L'],
        0.0
    )
    del _df['H-pH'], _df['pL-L']

    # +- DMI
    _df['S+DM'] = _df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    _df['S-DM'] = _df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    _df['+DMI'] = (_df['S+DM']/_df['ATR'])*100
    _df['-DMI'] = (_df['S-DM']/_df['ATR'])*100
    del _df['S+DM'], _df['S-DM']

    # ADX
    _df['DX'] = (np.abs(_df['+DMI'] - _df['-DMI'])/(_df['+DMI'] + _df['-DMI']))*100
    _df['ADX'] = _df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del _df['DX'], _df['ATR'], _df['TR'], _df['-DX'], _df['+DX'], _df['+DMI'], _df['-DMI']

    return _df

# https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe
def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

# https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe
def RSI(data: pd.DataFrame, period = 14) -> pd.DataFrame:
    _df = data.copy()
    _df['change'] = _df['close'].diff()
    _df['gain'] = _df.change.mask(_df.change < 0, 0.0)
    _df['loss'] = -_df.change.mask(_df.change > 0, -0.0)
    _df['avg_gain'] = rma(_df.gain[period+1:].to_numpy(), period, np.nansum(_df.gain.to_numpy()[:period+1])/period)
    _df['avg_loss'] = rma(_df.loss[period+1:].to_numpy(), period, np.nansum(_df.loss.to_numpy()[:period+1])/period)
    _df['rs'] = _df.avg_gain / _df.avg_loss
    _df['RSI'] = 100 - (100 / (1 + _df.rs))

    del _df['change'], _df['gain'], _df['loss'], _df['avg_gain'], _df['avg_loss'], _df['rs']
    return _df

def find_grid(grids: list, current_price: float) -> int:
    # binary search
    # input current price
    # returns the index of the next closest sell
    # 1     2     3     4     5      6
    # (0-1] (1,3] (3,5] (5,9] (9,11] (11,infty)
    # find_grid([1,3,5,9,11,15], 1.1) -> 2
    # find_grid([1,3,5,9,11,15], 1.0) -> 1

    # print('in function, current', current_price)

    if current_price <= grids[0].limitbuy:
        return 0
    if current_price > grids[-1].limitbuy:
        return len(grids)
    
    grids = [Grid(0, 0, grids[0].limitbuy)] + grids[:] + [Grid(len(grids), grids[-1].limitsell, np.inf)]
    # print([f'({x.limitbuy}, {x.limitsell}]' for x in grids])

    low = 0
    upper = len(grids) - 2
    while low <= upper:
        mid = (low + upper) // 2
        # print('mid', mid)
        if grids[mid].limitbuy < current_price and current_price <= grids[mid+1].limitbuy: #若搜尋值等於中間的值，則回傳
            return min(mid, len(grids)-2)
            # return mid-1
        elif grids[mid].limitbuy < current_price and grids[mid+1].limitbuy < current_price: #若搜尋值比中間的值大，將中間索引+1，取右半
            low = mid + 1
        elif current_price < grids[mid].limitbuy: #若搜尋值比中間的值小，將中間索引+1，取左半
            upper = mid - 1
        elif current_price == grids[mid].limitbuy:
            return mid-1
    return -10e8

def round_decimals_down(number: float, decimals: int=2) -> float:
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_to_min_tick(value: float) -> float:
    if   value>1000:  return round_decimals_down(value, 2)
    elif value>10:    return round_decimals_down(value, 3)
    elif value>0.1:   return round_decimals_down(value, 4)
    else:             return round_decimals_down(value, 8)

def set_arithmetic_grid(lower: float, upper: float, num: int, TX_FEE: float=0.0005) -> list:
    """upper: upper bound / lower: lower bound / num: number of grids"""
    if lower >= upper: raise BadGridBoundsError(lower, upper)
    grids = []
    gap = (upper - lower) / (num)
    for i in range(num):
        b = round_to_min_tick(lower + i * gap)
        s = round_to_min_tick(lower + (i+1) * gap)
        # Grid(index: int, limitbuy: float, limitsell: float)
        grids.append( Grid(i, b, s) )

    return grids

def set_geometric_grid(lower: float, upper: float, num: int, TX_FEE: float=0.0005) -> list:
    if lower >= upper: raise BadGridBoundsError(lower, upper)
    grids = []
    percent_per_grid = (upper / lower) ** (1 / num)
    for i in range(num):
        # Grid(index: int, limitbuy: float, limitsell: float)
        b = round_to_min_tick(lower * (percent_per_grid) **  i   )
        s = round_to_min_tick(lower * (percent_per_grid) ** (i+1))
        grids.append( Grid(i, b, s) )

    return grids

def set_modified_grid(lower: float, lower_num: int, upper: float, upper_num: int, start: float, TX_FEE: float=0.0005) -> list:
    lower_num -= 1
    
    if lower >= upper: raise BadGridBoundsError(lower, upper)
    upper_percent_per_grid = (upper / start) ** (1 / upper_num)
    lower_percent_per_grid = (start / lower) ** (1 / lower_num)
    grids = []
    cost = 0

    for i in range(lower_num, -1, -1):
        # print(i)
        # Grid(index: int, limitbuy: float, limitsell: float)
        b = round_to_min_tick(start * (lower_percent_per_grid) ** (-i-1) )
        s = round_to_min_tick(start * (lower_percent_per_grid) ** (-i))
        grids.append( Grid(lower_num-i, b, s) )
        cost += b / ((1-TXFEE)**2)

    cost += start * upper_num / ((1-TXFEE)**2)
    for i in range(upper_num-1, -1, -1):
        # Grid(index: int, limitbuy: float, limitsell: float)
        b = round_to_min_tick(upper - start * ((upper_percent_per_grid) ** (i+1) - 1) )
        if i == upper_num-1:
            b = start
        s = round_to_min_tick(upper - start * ((upper_percent_per_grid) ** (i)  - 1) )
        grids.append( Grid(lower_num+upper_num-i, b, s) )
    
    return grids, cost




# ===========================================================================

TXFEE = 0.0005
mode  = ['GEOMETRIC', 'ARITHMETIC', 'MODIFIED'][2]
# myaccount = Account(100000)

# df  = pd.read_csv('./csv/exp.csv')
df  = pd.read_csv('./csv/btc.csv')
df.timestamp = pd.to_datetime(df.timestamp)

# df = df[df.timestamp >= datetime.datetime(2021,1,1)].reset_index(drop=True)
print(df)

if   mode == 'GEOMETRIC' : grids = set_geometric_grid(6000, 80000, 500)
elif mode == 'ARITHMETIC': grids = set_arithmetic_grid(6000, 80000, 500)
elif mode == 'MODIFIED': grids, cost = set_modified_grid(6000, 69, 80000, 431, 16322)


@timer
def main():
    myaccount = Account(100000)

    grid_cost_sum  = 0
    each_grid_coin = 0

    for minute in df.itertuples():
        if minute.Index==0:
            price = minute.close
            # price = 5800.29
            lowestBuyIndex = find_grid(grids, price)

            if mode == 'MODIFIED':
                grid_cost_sum = cost
            else:
                # print('lbi', lowestBuyIndex)

                mktbuyamt = len(grids) - lowestBuyIndex  # number of grids to buy at the 1st minute
                grid_cost_sum += (mktbuyamt * price) / ((1-TXFEE)**2)

                n = lowestBuyIndex  # number of grids that cant be bought at the 1st minute
                if mode == 'GEOMETRIC':
                    a0 = grids[0].limitbuy
                    r  = grids[0].limitsell / grids[0].limitbuy
                    grid_cost_sum += (a0 * (r**n - 1) / (r-1)) / ((1-TXFEE)**2)
                elif mode == 'ARITHMETIC':
                    a0 = grids[0].limitbuy
                    d  = grids[0].limitsell - grids[0].limitbuy
                    grid_cost_sum += ((n/2) * ( 2*a0 + (n-1)*d )) / ((1-TXFEE)**2)

            each_grid_coin = round_decimals_down(myaccount.balance / grid_cost_sum, 5)
            # print('each grid coin', each_grid_coin)

            # indices for the 2nd minute. may exceed index range!
            # [ -1; 0, ..., len-1; len]
            myaccount.next_buy_index = lowestBuyIndex - 1
            myaccount.next_sell_index = lowestBuyIndex

            # will not enter loof if starting price is higher than all LIMITs.
            for buyidx in range(lowestBuyIndex, len(grids)):
                myaccount.balance -= grids[buyidx].buy(price, each_grid_coin, 'MKT', TXFEE)

            # print(f'\nMinute 0 ${price}')
            # print(f'next buy: {myaccount.next_buy_index} ${grids[myaccount.next_buy_index].limitbuy} > ${grids[myaccount.next_buy_index].limitsell}')
            # print(f'next sell: {myaccount.next_sell_index} ${grids[myaccount.next_sell_index].limitbuy} > ${grids[myaccount.next_sell_index].limitsell}')

        else:
            # 2nd minute and onward
            # print(f'\nMinute {minute.Index}')
            if minute.open > minute.close:
                # going down this minute ==> can only BUY
                buyidx = myaccount.next_buy_index
                if buyidx == -1:
                    # print('B Lowest Bought')
                    continue
                
                while grids[buyidx].limitbuy >= minute.close:
                    myaccount.balance -= grids[buyidx].buy(minute.close, each_grid_coin, 'LIM', TXFEE)
                    # print(f'Bought {buyidx} at ${grids[buyidx].buyprice}')

                    myaccount.next_sell_index -= 1
            
                    buyidx -= 1
                    if buyidx == -1:
                        # print(f'B Lowest Bought')
                        # print(f'B next sell: {myaccount.next_sell_index} ${grids[myaccount.next_sell_index].limitbuy} > ${grids[myaccount.next_sell_index].limitsell}')
                        break
                    
                    # print(f'B next buy: {buyidx} ${grids[buyidx].limitbuy} > ${grids[buyidx].limitsell}')
                    # print(f'B next sell: {myaccount.next_sell_index} ${grids[myaccount.next_sell_index].limitbuy} > ${grids[myaccount.next_sell_index].limitsell}')

                myaccount.next_buy_index = buyidx

            else:
                # going up this minute ==> can only SELL
                sellidx = myaccount.next_sell_index
                if sellidx == len(grids):
                    # print(f'S Highest Sold')
                    continue

                while grids[sellidx].limitsell < minute.close:
                    cash_returned, profit = grids[sellidx].sell(minute.close, 'LIM', TXFEE)
                    myaccount.gridprofit += profit
                    myaccount.balance += cash_returned
                    # print(f'Sold {sellidx} at ${grids[sellidx].limitsell}, earn {profit}')
                    
                    myaccount.next_buy_index += 1
                    sellidx += 1

                    if sellidx == len(grids):
                        # print(f'S next buy: {myaccount.next_buy_index} ${grids[myaccount.next_buy_index].limitbuy} > ${grids[myaccount.next_buy_index].limitsell}')
                        # print(f'S Highest Sold')
                        break

                    # print(f'S next buy: {myaccount.next_buy_index} ${grids[myaccount.next_buy_index].limitbuy} > ${grids[myaccount.next_buy_index].limitsell}')
                    # print(f'S next sell: {sellidx} ${grids[sellidx].limitbuy} > ${grids[sellidx].limitsell}')

                myaccount.next_sell_index = sellidx



    coin_asset = 0
    unrealized_profit = 0
    for grid in grids:
        # print(grid.index, grid.quantity, grid.buyprice, minute.close - grid.buyprice)
        coin_asset += grid.quantity * minute.close * (1-TXFEE)
        unrealized_profit += grid.quantity * (minute.close * (1-TXFEE) - grid.buyprice / (1-TXFEE)) 

    print('\nFinal Stats\n')
    print(f'Cash Balance ${round(myaccount.balance, 2)}')
    print(f'Total Equity ${round(coin_asset + myaccount.balance, 2)}')
    print(f'Total Return {round(100*((coin_asset + myaccount.balance) / myaccount.initial_balance - 1) , 2)}%' )
    print(f'Unrl. Loss ${round(unrealized_profit, 2)}')
    print(f'Unrl. Return {round(100*(unrealized_profit / myaccount.initial_balance), 2)}%')
    print(f'Grid Profit ${round(myaccount.gridprofit, 2)}')
    print(f'Grid Return {round(100*(myaccount.gridprofit / myaccount.initial_balance) , 2)}%')
    # print('\n', myaccount.next_buy_index, myaccount.next_sell_index)


main()


