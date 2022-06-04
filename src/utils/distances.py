from scipy.spatial import distance
import numpy as np

MAX_CLASS = 13 #maksymalne id klasy


def compute_distances(x, embedding, method=distance.euclidean):
    distances = np.zeros((embedding.shape[0]))
    for i in range(embedding.shape[0]):
        distances[i] = method(embedding[i], x)
    return distances
