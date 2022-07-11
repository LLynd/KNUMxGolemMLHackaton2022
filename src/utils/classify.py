from scipy.spatial import distance
import numpy as np
import numpy.typing as npt


def compute_distances(x: float, embedding: npt.ArrayLike,
                      method=distance.cosine) -> list:
    '''
    A function computing a list of distances between . Returns..
    x =
    embedding =
    method =

    '''
    distances = np.zeros((embedding.shape[0]))
    for i in range(embedding.shape[0]):
        distances[i] = method(embedding[i], x)
    return distances


def classify(distances, classes):
    '''
    A function that classifies samples based on . Returns
    distances - list of distances between  ; same dimensions as classes.
    classes = list of labels
    :type distances: list[float]
    :type classes: list[str]
    '''
    weights = {cls: [] for cls in classes}
    for i in range(len(distances)):
        weights[classes[i]].append(distances[i])

    m = max(sum(list(map(lambda x: pow(x, -1), scores)))/len(scores) for scores
            in weights.values())
    for cls in weights.keys():
        lm = sum(list(map(lambda x: pow(x, -1),
                          weights[cls])))/len(weights[cls])
        if lm == m:
            return cls
