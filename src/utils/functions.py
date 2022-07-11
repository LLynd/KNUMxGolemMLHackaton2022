from scipy.spatial import distance
import numpy as np
import pandas as pd
import json


def compute_distances(x, embedding, method=distance.cosine):
    '''
    opis
    '''
    distances = np.zeros((embedding.shape[0]))
    for i in range(embedding.shape[0]):
        distances[i] = method(embedding[i], x)
    return distances


def load_data(mode):
    '''
    opis
    '''
    if mode == 'TRAIN':
        json_path = '../../data/reference_images_part2.json'
        images_path = '../../data/reference_images_part2/'
    elif mode == 'VAL':
        json_path = '../../data/images_part2_test_public.json'
        images_path = '../../data/images_part2_test/'
    else:
        raise ValueError('usupported mode')

    with open(json_path) as json_data:
        data = json.load(json_data)

    images = pd.DataFrame(data['images'])
    annotations = pd.DataFrame(data['annotations'])
    categories = pd.DataFrame(data['categories'])
    df = pd.DataFrame()

    X = []
    y = []
    y_desc = []
    occluded = []
    bboxes = []
    im_ids = []

    for instance in data['annotations']:
        im_id = instance['image_id']
        bbox = instance['bbox']
        y.append(instance['category_id'])
        bboxes.append(np.asarray(bbox).astype('int64'))
        # print(images.loc[images['id']==im_id]['file_name'])
        im_ids.append(images.loc[images['id'] == im_id]['file_name'].values[0])
        # y_desc.append(categories.loc[categories['id']==instance['category_id']]['name'].values[0])
        if mode == 'TRAIN':
            occluded.append(False)
        elif mode == 'VAL':
            occluded.append(instance['occluded'])

    df['bbox'] = bboxes
    df['y'] = y
    # df['desc'] = y_desc
    df['im_id'] = im_ids
    df['occ'] = occluded

    return df


def classify(distances, classes):
    '''
    same lengths
    distances - np.array
    classes - np.array
    returns: class
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
