from objfun import ObjFun

from sklearn.cluster import KMeans
from sklearn import metrics

import networkx as nx
import numpy as np
import pandas as pd

# BiasedRandomWalk = node2vec realised in stellargraph library
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph

from gensim.models import Word2Vec


class N2V(ObjFun):
    """
    general n2v run model
    """

    def __init__(self, edges_path, lables_path):
        """
        Hard-coded initialization
        """
        fstar = 1
        a = 0.125 # p, q left bound
        b = 4.125 # p, q right bound
        graph, labels = self.read_data(edges_path, lables_path)
        rw = BiasedRandomWalk(StellarGraph.from_networkx(graph))

        super().__init__(fstar, a, b, graph, labels, rw)

    def set_b(self,b):
        self.b=b

    def get_b(self):
        return self.b

    def read_data(self, edges_path, lables_path):
        """
        data read from csv files and construction of a graph
        """
        edges = pd.read_csv(edges_path, header=None, names=('from', 'to'))
        labels = pd.read_csv(lables_path, header=None,
                             names=('node', 'label'))

        graph = nx.Graph()
        graph.add_nodes_from(labels['node'])

        for i, row in edges.iterrows():
            graph.add_edge(row['from'], row['to'], weight=1)

        return graph, labels

    def generate_point(self):
        """
        Random point generator
        :return: random point from the domain
        """
        # return round(np.random.uniform(0.125, 4.125),2)
        return np.random.uniform(0.125, 4.125)

    def get_neighborhood(self, x, d):
        """
        Solution neighborhood generating function
        :param x: point
        :param d: diameter of the neighbourhood
        :return: list of points in the neighborhood of the x
        """
        left = [x for x in np.linspace(x-0.125, x -d,2) if x >= self.a]
        right = [x for x in np.linspace(x+0.125, x + d,2) if x < self.b]
        if np.size(left) == 0:
            return right
        elif np.size(right) == 0:
            return left
        else:
            return np.concatenate((left, right))

    def evaluate(self, p=1, q=1, num_walks=10, len_walks=80, window=10):
        """
        Objective function evaluating function
        The default values of parameters are the same as in the original paper
        :param p: (unormalised) probability, 1/p, of returning to source node
        :param q: (unormalised) probability, 1/q, for moving away from source node
        :param num_walks: number of random walks per root node
        :param len_walks: maximum length of a random walk
        :param window: point
        :return: objective function value
        """
        # print('random walks generation')
        weighted_walks = self.rw.run(
            nodes=self.graph.nodes(),  # root nodes
            length=len_walks,
            n=num_walks,
            p=p,
            q=q,
            weighted=False,  # for weighted random walks
            seed=42  # random seed fixed for reproducibility
        )

        # converting integers into string, otherwise Word2Vec throws an error
        weighted_walks = [[str(j) for j in i] for i in weighted_walks]
        # print('word2vec')
        model = Word2Vec(weighted_walks, size=128, window=window, min_count=0, sg=1, workers=1, iter=1)
        # print('known_labels calculation')
        known_labels = []
        for i in model.wv.index2word:
            known_labels.append(self.labels.loc[int(i)]['label'])
        # print('clusterisation')
        n_clusters = len(self.labels['label'].unique())
        km = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(model.wv.vectors)

        score = metrics.adjusted_rand_score(known_labels,km)

        return score
