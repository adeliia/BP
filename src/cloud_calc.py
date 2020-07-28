# Import external libraries
import pandas as pd
from tqdm.notebook import tqdm

import seaborn as sns
sns.set()
from objfun_node2vec import N2V
from heur_go import GeneticOptimization
from datetime import datetime

NUM_RUNS = 5 #1000
maxeval = 5 #1000

def experiment_go(of, maxeval, num_runs, N):
    results = []
    heur_name = 'GO_{}'.format(N)
    for i in range(num_runs):
        result = GeneticOptimization(of, maxeval, N=N).search()
        print('result ', result)
        result['run'] = i
        result['heur'] = heur_name
        result['N'] = N
        results.append(result)

        # write results to csv file
        now = datetime.now()
        current_time = now.strftime('%Y%m%d%H%M')
        res = pd.DataFrame(results, columns=['heur', 'run', 'N', 'best_x', 'best_y', 'neval'])
        res.to_csv('../results/' + current_time + '_one_eighth_GO_' + str(N) + '.csv')

    return res

def run():
    # initialization
    n2v = N2V('../data/emails/edges.csv',
              '../data/emails/labels.csv')

    results = pd.DataFrame()
    for N in [2, 4, 8, 10]:
        print('N ', N)
        res = experiment_go(of=n2v, maxeval=maxeval, num_runs=NUM_RUNS, N=N)
        results = pd.concat([results, res], axis=0)


run()