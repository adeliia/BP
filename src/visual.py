import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
seed = 42

def to2d(embedding_clusters, perplexity):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape

    tsne_model_en_2d = TSNE(perplexity=perplexity, n_components=2, random_state=seed)
    embeddings_en_2d = np.array(
        tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    return embeddings_en_2d


colors_dic = {
    0: 'red',
    1: 'orange',
    2: 'black',
    3: 'green',
    4: 'blue'
}


def assign_colors(pure_model_nodes, labels):
    colors = []
    for i in pure_model_nodes:
        colors.append(colors_dic[labels.loc[int(i)]['label']])
    return colors


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, colors, a, filename=None):
    plt.figure(figsize=(10, 7))

    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=colors_dic[label], alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=1, xy=(x[i] + 2, y[i] + 2), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=11)
    plt.legend(loc=4, prop={'size': 12})
    plt.title(title)
    plt.grid(True)
    plt.show()