# genetic-algorithm-for-portfolio-optimisation

Markowitz (1952) proposed the foundation for modern portfolio theory with mean-variance portfolio optimisation. Providing a method for investors to calculate how best to allocate and weight their capital to individual shares based upon their comovements. 

Genetic Algoritms are optimisation algorithms which replicates the processes in Charles Darwins theory of natural evolution.

Thus, a GA can be applied to the portfolio optimisation problem. This python script takes some user inputted stock tickers and returns the optimal weights. The weights are evaluated by the objective function which is the Sharpe Ratio. The optimal portfolio will be the one which offers the best Sharpe Ratio, meaning the best expected return per unit of risk.

The GA is real-value encoded and utilises uniform crossover and boundary mutation with selection based on the roulette wheel method as well as employing elitism. 
