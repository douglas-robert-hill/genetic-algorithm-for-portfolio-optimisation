

def get_data(tickers):
    portfolio = pd.DataFrame()
    
    for t in tickers:
        portfolio[t] = data.DataReader(t, data_source = 'yahoo', start='2019-02-01')['Adj Close']
        
    portfolio.columns = tickers
    returns = np.log(portfolio / portfolio.shift(1))
    
    port_return = np.array(returns.mean() * 252)
    port_risk = returns.cov()
    
    return portfolio, port_return, port_risk