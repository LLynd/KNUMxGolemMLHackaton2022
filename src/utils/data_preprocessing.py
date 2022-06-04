import numpy as np
import os
from PIL import Image


class Preprocessor:
    def __init__(self, mode):
        self.files = []
        self.mode = mode
        if mode == 'TRAIN':
            self.images_path = '../../data/reference_images_part1/'
            self.json_path = '../../data/reference_images_part1.json'
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
        return [self.files.index(i) for i in self.files]
    
    def preproces(self):
        idxs = self._get_file_index()
        mean = 0
        data_x = np.zeros((len(idxs),256,256,3))

        for file, i in zip(self.files, idxs):
            img = np.asarray(Image.open(file))
            img = img[:,:,:3]
            temp_df_shape = img.shape
            mean += np.mean(img)
            max_shape = np.max(temp_df_shape)
            max_shape_orient = np.where(temp_df_shape==max_shape) # zwróci 0 albo 1
            temp_arr = np.zeros((max_shape,max_shape,3)) ## mozna czymś wypełnić tło

            if max_shape_orient[0][0] == 0:
                t = (temp_df_shape[0] - temp_df_shape[1])//2
                temp_arr[:,t:t+temp_df_shape[1],:] = img

            elif max_shape_orient[0][0] == 1:
                t = (temp_df_shape[1] - temp_df_shape[0])//2
                temp_arr[t:t+temp_df_shape[0],:,:] = img
            
            data_x[i] = np.resize(temp_arr,(256,256,3))   
        data_x = np.where(data_x==0, mean/(i+1), data_x)

        for i in range(data_x.shape[0]):
            tmp = data_x[i]
            np.save(f'../../data/preprocessed/image{i}.npy', tmp)


prep = Preprocessor('TRAIN')
prep.preproces()
