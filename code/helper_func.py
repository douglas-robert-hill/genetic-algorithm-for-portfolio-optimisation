
def normalise(generation, assets):
    for i in range(0, len(generation)):
        generation[i][0:assets] /= np.sum(generation[i][0:assets])
    return generation



def next_gen(elites, children, no_cross_parents):
    weights = np.vstack((elites, children, no_cross_parents))
    return weights 



def optimal_solution(generations, assets):
    optimal_weights = generations[generations[:, (assets + 1)].argsort()]
    return optimal_weights[0]



def avg_gen_result(weights, n_assets):
    average = round(np.mean(weights[:, n_assets]), 2)
    return average
