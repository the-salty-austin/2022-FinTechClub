def set_arithmetic_grid(upper, lower, num, TX_FEE=0.0005):
    '''upper: upper bound / lower: lower bound / num: number of grids'''
    grids = []
    gap = (upper-lower) / (num-1)
    for i in range(num):
        grids.append( lower + i*gap )

    grid_profits = [0]*num
    for i in range(num):
        if i==0: grid_profits[i]=0
        else: grid_profits[i] = (grids[i]/grids[i-1] - 1)*(1-TX_FEE)

    return grids, grid_profits


def set_geometric_grid(upper, lower, num, TX_FEE=0.0005):
    grids = []
    percent_per_grid = (upper/lower)**(1/(num-1))
    for i in range(num):
        grids.append( lower*(percent_per_grid)**i )

    grid_profits = [0]*num
    for i in range(num):
        if i==0: grid_profits[i]=0
        else: grid_profits[i] = (grids[i]/grids[i-1] - 1)*(1-TX_FEE)
    
    return grids, grid_profits


def asset_evaluation(daily_tx_cnts, status, cash, grid_profit, cur_price, INVEST=1000, NUM=200, TX_FEE=0.0005):
    asset_value = 0
    for i in range(NUM-1):
        # print(status[i]['holding'], cur_price)
        asset_value += status[i]['holding']*cur_price
    # total = asset_value + cash
    
    grid_profit_perc = round( 100*(grid_profit/INVEST) ,2)

    unrealized_profit = 0
    for i in range(NUM-1):
        p = status[i]['bought_at']
        if p == -1: continue
        # print(status[i]['holding'],cur_price-p)
        unrealized_profit += status[i]['holding']*(cur_price-p/(1-TX_FEE))
    unrealized_profit_perc = round( 100*(unrealized_profit/INVEST ),2)

    total = cash + asset_value
    # total = INVEST + grid_profit + unrealized_profit
    total_perc = round(100*( total/INVEST )-100,2)

    print(f'\nDays elapsed: {len(daily_tx_cnts)}')
    print('---------|-------------|---------')
    print('{:>9}|{:>13}|{:>9}'.format('Grid', 'Unrealized', 'Total'))
    print('---------|-------------|---------')
    print('{:>9}|{:>13}|{:>9}'.format( round(grid_profit,2) , round(unrealized_profit,2) , round(total,2) ))
    print('{:>9}|{:>13}|{:>9}'.format(str(round( 100*(grid_profit/INVEST) ,2))+'%', str(round( 100*(unrealized_profit/INVEST ),2))+'%', str(round(100*( total/INVEST )-100,2))+'%'))
    print('---------|-------------|---------')
    print('{:>9}|{:>13}|{:>9}'.format('Grid APR', 'Unr. APR', 'Tot. APR'))
    print('---------|-------------|---------')
    print('{:>9}|{:>13}|{:>9}'.format(str(round( 100*(grid_profit/INVEST)*(365/len(daily_tx_cnts)) ,2))+'%', str(round( 100*(unrealized_profit/INVEST )*(365/len(daily_tx_cnts)),2))+'%', str(round( 100*( total/INVEST -1 )*(365/len(daily_tx_cnts)) ,2 ))+'%'))
    print('---------|-------------|---------')
    print(f'Tx count: {sum(daily_tx_cnts)} / avg: {sum(daily_tx_cnts)/len(daily_tx_cnts)}\n')

    return daily_tx_cnts, grid_profit_perc, unrealized_profit_perc, total_perc