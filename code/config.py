
import numpy as np
import pandas as pd
import random
import pandas_datareader.data as web

tickers = ['AAPL', 'TSLA', 'MSFT']
population = 150
risk_free_rate = 2
generations = 40
crossover_rate = 0.4
mutation_rate = 0.01
elite_rate = 0.25


