from utility import *
from download import *
from utility import timer
import datetime

import pandas as pd


class NoCoinToSell(Exception):
    pass


class NoCashToBuy(Exception):
    pass


class AlreadyBought(Exception):
    pass


class UndefinedMode(Exception):
    pass


@timer
def backtest(
    df, UPPER, LOWER, grid_mode, NUM=200, TX_FEE=0.0005, INVEST=1000, show_tx=True
):

    # pp = pprint.PrettyPrinter(indent=2)

    df.timestamp = pd.to_datetime(df.timestamp)

    if grid_mode == "arithmetic":
        grids, grid_profits = set_arithmetic_grid(UPPER, LOWER, NUM)
    elif grid_mode == "geometric":
        grids, grid_profits = set_geometric_grid(UPPER, LOWER, NUM)
    else:
        raise UndefinedMode

    # initializing the grid trade bot
    # say, current price is 32, and the grids are [20,25,30,35,40,45]
    #
    # if a grid's sell price is higher than the current price,
    # we will buy right now and wait for the price to rise.
    # So in this case, we'll buy the ones at 32, and wait to sell at 35,40,45
    #
    # On the contrary, if grid's sell price is lower,
    # will not buy now. will wait the price to drop and then buy.
    # in this case, we'll wait for the price to drop to 30,25,20
    cur_price = df.close.iloc[0]  # current price is 32
    grid_sum = 0  # total cost of all grids
    for i in range(len(grids)):
        if i == NUM - 1:
            continue
        if grids[i + 1] > cur_price:
            # if sell price of the grid > cur_price
            # cost is cur_price*(1 + transaction fee)
            grid_sum += cur_price * (1 + TX_FEE)
        else:
            # if sell price of the grid < cur_price
            # cost is the buy price of the grid
            grid_sum += grids[i] * (1 + TX_FEE)

    each_grid_coin = (INVEST / grid_sum) * (1 - TX_FEE)
    cash = INVEST

    # a dictionary containing the current status of each grid
    status = {}
    for i in range(len(grids)):
        if i == NUM - 1:
            continue
        # grids[i+1] is the sell price of grid i.
        if grids[i + 1] <= cur_price:
            # current price >= grid sell price
            # this means: don't buy.
            status[i] = {
                "buy_price": grids[i],
                "sell_price": grids[i + 1],
                "status": "cash",
                "bought_at": -1,
                "holding": 0,
            }
        else:
            # current price < grid sell price
            # this means: buy, wait to sell.
            status[i] = {
                "buy_price": grids[i],
                "sell_price": grids[i + 1],
                "status": "coin",
                "bought_at": cur_price,
                "holding": each_grid_coin,
            }
            cash -= (each_grid_coin / (1 - TX_FEE)) * cur_price

    day_sell_tx_cnt, day_buy_tx_cnt = 0, 0
    daily_sell_tx_cnts, daily_buy_tx_cnts = [], []
    daily_grid_profit = 0
    daily_grid_profits = []
    prev_grid = -1

    # iterating over each minute
    prev_date = df.timestamp.iloc[1].date()
    last_tx_i = -1
    first_tx_executed = False
    for minute_idx, row in df.iterrows():
        if minute_idx == 0:
            continue

        cur_price = df.close.iloc[
            minute_idx - 1
        ]  # previous closing price as opening price
        if cur_price >= row.close:
            high = cur_price
            low = row.close
        else:
            high = row.close
            low = cur_price

        # among all the grids, how many are buys/sells?
        # e.g. current price is 195, all grids with prices >=195 are SELLing ones. <195 are BUYing ones.
        buy = []
        sell = []
        for j, grid in enumerate(grids):
            if grid <= cur_price:
                # if grid <= row.close:
                if j == NUM - 1:
                    continue  # top grid: only sell, no buy
                buy.append(j)
            elif grid >= cur_price:
                # elif grid > row.close:
                if j == 0:
                    continue  # bottom grid: no sell, only buy
                sell.append(j)

        # print(f'[{row.timestamp}] | Highest Buy: ${round(grids[buy[-1]],1)} ({buy[-1]}) | ${row.close} | Lowest Sell: ${round(grids[sell[0]],2)} ({sell[0]})')

        # count how many grids are in this minute's price range (low~high)
        cnt = 0
        tx_idx = []
        for i, grid in enumerate(grids):
            if low <= grid <= high:
                tx_idx.append(i)
                cnt += 1

        # if price drops this minute, invert trading sequence
        # original sequence is to trade from lower grids to higher ones
        if cur_price - row.close > 0:
            tx_idx = tx_idx[::-1]

        if cnt > 0:
            # simulate to trade, according to the sequence above
            for tx_i in tx_idx:
                # print(tx_i,prev_grid)
                if tx_i == prev_grid:
                    continue
                else:
                    prev_grid = tx_i

                """
                status[i]: {
                        'buy_price': grids[i],
                        'sell_price': grids[i+1],
                        'status': 'coin',
                        'bought_at': cur_price,
                        'holding': each_grid_coin
                    }
                """
                if tx_i in sell:
                    # sell coin, get USDT
                    if show_tx:
                        print(f"{row.timestamp}, {tx_i} SOLD, ${round(grids[tx_i],3)}")

                    if status[tx_i - 1]["status"] == "cash" and first_tx_executed:
                        raise NoCoinToSell
                    day_sell_tx_cnt += 1
                    cash += (
                        status[tx_i - 1]["holding"] * status[tx_i - 1]["sell_price"]
                    ) * (1 - TX_FEE)
                    daily_grid_profit += status[tx_i - 1]["holding"] * (
                        (1 - TX_FEE) * status[tx_i - 1]["sell_price"]
                        - status[tx_i - 1]["bought_at"] * (1 + TX_FEE)
                    )

                    status[tx_i - 1]["status"] = "cash"
                    status[tx_i - 1]["bought_at"] = -1
                    status[tx_i - 1]["holding"] = 0
                    last_tx_i = tx_i - 1
                    first_tx_executed = True

                elif tx_i in buy:
                    # buy coin, get coin
                    if show_tx:
                        print(
                            f"{row.timestamp}, {tx_i} BOUGHT, ${round(grids[tx_i],3)}"
                        )

                    if cash <= 0:
                        print(cash)
                        raise NoCashToBuy
                    if status[tx_i]["status"] == "coin" and first_tx_executed:
                        # pp.pprint(status)
                        raise AlreadyBought

                    day_buy_tx_cnt += 1
                    cash -= each_grid_coin * grids[tx_i] * (1 + TX_FEE)

                    status[tx_i]["status"] = "coin"
                    status[tx_i]["bought_at"] = grids[tx_i]
                    status[tx_i]["holding"] = each_grid_coin
                    last_tx_i = tx_i
                    first_tx_executed = True

        # print(minute_idx+1)
        if (
            minute_idx == len(df) - 1
            or df.timestamp.iloc[minute_idx + 1].date() != prev_date
        ):
            # print(day_sell_tx_cnt, day_buy_tx_cnt)
            if minute_idx < len(df) - 1:
                prev_date = df.timestamp.iloc[minute_idx + 1].date()  # tomorrow
            daily_sell_tx_cnts.append(day_sell_tx_cnt)
            daily_buy_tx_cnts.append(day_buy_tx_cnt)
            daily_grid_profits.append(daily_grid_profit)
            day_buy_tx_cnt = 0
            day_sell_tx_cnt = 0
            daily_grid_profit = 0

    asset_evaluation(
        daily_sell_tx_cnts,
        status,
        cash,
        sum(daily_grid_profits),
        df.close.iloc[-1],
        INVEST=INVEST,
        NUM=NUM,
    )

    return daily_sell_tx_cnts, cash, daily_grid_profits


if __name__ == "__main__":
    # df = get_data_since(
    #     "BTC",
    #     datetime.datetime(2022, 11, 1, 0, 0),
    #     datetime.datetime(2022, 11, 2, 12, 30),
    # )
    df = pd.read_csv('./csv/btc.csv')
    backtest(df, 68000, 7000, "geometric", NUM=500, TX_FEE=0.0005, INVEST=100000, show_tx=False)
