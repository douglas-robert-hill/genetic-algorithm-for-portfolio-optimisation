import data
import genetic_algo
import config

optimal_weights = genetic_algorithm(tickers, risk_free_rate, population, generations, crossover_rate, mutation_rate, elite_rate)
optimal_weights
