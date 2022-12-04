import pandas as pd


df = pd.read_csv("./csv/btc.csv").tail(200000)

CAPITAL = 1000
INC = 0.006  # when price drops 2%, increase position size.
TP = 0.006  # when price rebounces, sell when profit of total holdings is +1%
MULT = 1.5 # when price drops [INC]%, buy 1>2>4>8>16 units.
LIMIT = 8  # buy at most 8 times
STOP = 0.012  # holding stop loss at 10%

TXFEE = 0.0000

def target_buys(current_price, cash):
    totalSlice = ( MULT**(LIMIT) - 1 ) / ( MULT - 1 )
    limit_orders = []
    for i in range(LIMIT):
        curTotSlice = ( MULT**(i+1) - 1 ) / ( MULT - 1 )
        limit_orders.append( {
            "index": i+1,
            "cash" : cash*(MULT**i/totalSlice) ,
            "limit": current_price * (1-INC)**i ,
            "weighted_cost": (1/curTotSlice) * current_price * (( MULT*(1-INC) )**(i+1) - 1) / (MULT*(1-INC) - 1)
        } )
    # for x in limit_orders:
    #     print(x)
    return limit_orders

last_bought = 0
limit_orders = []
target_sell = 0.0
coinsHeld = 0.0

cash = CAPITAL
rounds = 0
stopCnt = 0

netAsset = []

for row in df.itertuples():
    if cash <= 0 and last_bought<LIMIT:
        print("BROKE. No more cash.")
        break
    if cash + coinsHeld*row.close <= CAPITAL*0.5:
        print("50% assets gone. Stop.")
        break
    
    if last_bought == 0:
        limit_orders = target_buys(row.close, cash)

        last_bought = limit_orders[0]["index"]
        coinsHeld += limit_orders[0]["cash"]*(1-TXFEE) / row.close
        cash -= limit_orders[0]["cash"]
        target_sell = row.close * (1+TP)

        rounds += 1
        print(f"\nTime {row.timestamp}")
        print(f"Starting Round {rounds}.")
        print(f"Cash ${round(cash,2)}, Holdings {coinsHeld}")

    elif LIMIT > last_bought > 0:
        if row.high >= target_sell:
            # sell all holdings
            cash += (coinsHeld * target_sell) * (1-TXFEE)
            coinsHeld = 0.0
            last_bought = 0
            print(f"\nTime {row.timestamp}")
            print(f'Sold at ${target_sell}')
            print(f"Cash ${round(cash,2)}, Holdings {coinsHeld}")
        else:
            # check if can buy at lower price.
            for i in range(last_bought, LIMIT):
                if row.low <= limit_orders[i]["limit"]:
                    last_bought = limit_orders[i]["index"]
                    coinsHeld += limit_orders[i]["cash"] * (1-TXFEE) / row.close
                    cash -= limit_orders[i]["cash"]
                    target_sell = limit_orders[i]["weighted_cost"]
                    print(f"\nTime {row.timestamp}")
                    print(f'Buy. This ${round(limit_orders[i]["limit"],2)} / now low ${row.low}')
                    print(f"Cash ${round(cash,2)}, Holdings {coinsHeld}")

                elif row.high >= target_sell:
                    # weighted cost is low enough, so can sell right after buying low.
                    cash += (coinsHeld * target_sell) * (1-TXFEE)
                    coinsHeld = 0.0
                    last_bought = 0
                    print(f"\nTime {row.timestamp}")
                    print(f'Sold after buying low at ${target_sell}')
                    print(f"Cash ${round(cash,2)}, Holdings {coinsHeld}")
                    break
                
                else:
                    # print(f'No buy. Next ${round(limit_orders[i]["limit"],2)} / now low ${row.low}')
                    break
    
    elif last_bought==LIMIT:
        # print(f"\nTime {row.timestamp}")
        # print("Wait for rebounce.")
        stopLim = limit_orders[LIMIT-1]["weighted_cost"] * (1-STOP)
        if row.low >= stopLim:
            # sell all holdings
            cash += (coinsHeld * stopLim) * (1-TXFEE)
            coinsHeld = 0.0
            last_bought = 0

            stopCnt += 1
            print(f"\nTime {row.timestamp}")
            print(f'Stop Loss at ${stopLim}')
            print(f"Cash ${round(cash,2)}, Holdings {coinsHeld}")

    netAsset.append( cash + coinsHeld*row.close )

print("\n======== SUMMARY ========")
print(f"Peak Asset ${round(max(netAsset), 2)}")
print(f"Ending Asset ${round(netAsset[-1], 2)} ({100*round(netAsset[-1]/CAPITAL - 1, 4)}%)")
print(f"{rounds} rounds")
print(f"Stop Loss {stopCnt} times")

# df_dict = df.to_dict('records')
# for row in df_dict:
#     print(row)