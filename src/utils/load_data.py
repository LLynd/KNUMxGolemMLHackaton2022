import json
import pandas as pd
from PIL import Image
import numpy as np
import keras
import os
class LoadData:
    def __init__(self, mode):
        self.files = []
        self.mode = mode
        if mode == 'TRAIN':
            self.json_path = '../../data/reference_images_part1.json'
            self.images_path = '../../data/reference_images_part1/'
        elif mode == 'VAL':
            self.json_path = '../../data/images_part1_valid.json'
            self.images_path = '../../data/images_part1_valid/'
        else:
            raise ValueError('usupported mode')

        for file in os.listdir(self.images_path):
                    if file.endswith('.png'):
                        try:
                            self.files.append(os.path.join(self.images_path, file))
                        except FileNotFoundError as e:
                            print(file)

    def _get_file_index(self):
        return [i for i in self.files]

    def _load_data(self):

        with open(self.json_path) as json_data:
            data = json.load(json_data)
        images = pd.DataFrame(data['images'])
        annotations = pd.DataFrame(data['annotations'])
        categories = pd.DataFrame(data['categories'])

        self.df = pd.DataFrame()

        y = []
        y_desc = []

        for instance in data['annotations']:
            im_id = instance['image_id']
            bbox = instance['bbox']
            y.append(instance['category_id'])
            y_desc.append(categories.loc[categories['id']==instance['category_id']]['name'].values[0])

        self.df['y'] = y
        self.df['desc'] = y_desc

        return self.df

    def __getitem__(self, index):
        df = self._load_data()
        img = Image.open(self.files[index])
        X = np.asarray(img)
        y = df['y'].iloc[index]
        yield (X, y)
