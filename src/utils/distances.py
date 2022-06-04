from scipy.spatial import distance
import numpy as np

MAX_CLASS = 13 #maksymalne id klasy


def compute_distances(x, embedding, method=distance.euclidean):
    distances = np.zeros((embedding.shape[0]))
    for i in range(embedding.shape[0]):
        distances[i] = method(embedding[i], x)
    return distances


from scipy.spatial import distance

MAX_CLASS = 13

def compute_research_distances(df, embedding, method=distance.euclidean, mode='all'):
    if mode=='all':
        distances = np.zeros((embedding.shape[0],embedding.shape[0]))
        for i in range(embedding.shape[0]):
            for j in range(i,embedding.shape[0]):
                distances[i,j] = method(embedding[i],embedding[j])
                distances[j,i] = method(embedding[i],embedding[j])

    elif mode == 'in_group_mean' or mode == 'in_group_max':
        distances = np.zeros((MAX_CLASS+1))
        for i in range(MAX_CLASS+1):
            ids = df.loc[df['y']==i].index
            if len(ids)<2:
                pass
            else:
                in_class_dist = [method(embedding[ids[0]],embedding[ids[j]]) for j in range(1, len(ids))]
                if mode == 'in_group_mean':
                    distances[i] = np.mean(in_class_dist)
                elif mode == 'in_group_max':
                    distances[i] = max(in_class_dist)

    elif mode == 'inter_group_mean' or mode=='inter_group_min':
        distances = np.zeros((embedding.shape[0], MAX_CLASS+1))
        for i in range(embedding.shape[0]):
            for j in range(MAX_CLASS+1):
                ids = df.loc[df['y']==j].index

                if len(ids)<1:
                    pass
                elif i in ids:
                    distances[i,j] = np.NaN
                else:
                    dist_from_class = [method(embedding[i], embedding[ids[k]]) for k in range(len(ids))]
                    if mode=='inter_group_mean':
                        distances[i,j] = np.mean(dist_from_class)
                    elif mode=='inter_group_min':
                        distances[i,j] = np.min(dist_from_class)
    return distances
