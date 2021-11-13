
import helper_func
import ga_components

def genetic_algorithm(tickers, risk_free_rate, population, generations, crossover_rate, mutation_rate, elite_rate):
    weights, port_return, port_risk = get_data(tickers)
    weights, n_assets = generate_weights(weights, population)

    for i in range(0, generations):
        results = fitness_func(weights, port_return, port_risk, n_assets, risk_free_rate)
        
        elites, parents = elitism(elite_rate, results, n_assets)
        parents, no_cross_parents = selection(parents, n_assets)
        children = crossover(crossover_rate, parents, n_assets)
        children = mutation(mutation_rate, children, n_assets) 
        
        weights = next_gen(elites, children, no_cross_parents)
        
        avg_res = avg_gen_result(weights, n_assets)
        print('Generation', i, ': Average Sharpe Ratio of', avg_res, 'from', len(weights), 'chromosomes')
        
    opt_solution = optimal_solution(weights, n_assets)
    
    return opt_solution

    