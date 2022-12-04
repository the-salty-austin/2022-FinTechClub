import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('./csv/btc4h.csv')
df.timestamp = pd.to_datetime(df['timestamp'])
# df = df.set_index('timestamp')
# prices = df.close.to_numpy().ravel()
print( type(df.close) )
# prices = df['close'].to_numpy()

# print( prices )
# print( type(prices) )

# print( df )

# Monte-Carlo below

prices = df['close']
returns = prices.pct_change()
# returns = np.diff(prices) / prices[1:] * 100
daily_vol = returns.std()

last_price = prices.iloc[-1]

#Number of Simulations
num_simulations = 1000
num_days = 180

simulation_df = pd.DataFrame()

for x in range(num_simulations):
    count = 0
    
    price_series = []
    
    price = last_price * (1 + np.random.normal(0, daily_vol))
    price_series.append(price)
    
    for y in range(num_days):
        if count == num_days-1:
            break
        price = price_series[count] * (1 + np.random.normal(0, daily_vol))
        price_series.append(price)
        count += 1
    
    simulation_df[x] = price_series
    
fig = plt.figure()
fig.suptitle(f'Monte Carlo Simulation: BTC')
plt.plot(simulation_df)
plt.axhline(y = last_price, color = 'r', linestyle = '-')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()