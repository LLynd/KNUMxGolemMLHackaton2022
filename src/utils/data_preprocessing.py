import numpy as np
from skimage.transform import resize
import pandas as pd


def normalize(x):
    return x/255


def preprocessing(df):
    data_x = np.zeros((len(df),256,256,3))
    data_y = np.zeros((len(df),1))
    mean=0
    for i in range(len(df)):
        data_y[i] = df['y'][i]
        temp_df = np.array(df['X'][i][:,:,0:3])
        temp_df_shape = np.array(df['X'][i][:,:,0:3]).shape

        mean += np.mean(temp_df)
        max_shape = np.max(temp_df_shape)
        max_shape_orient = np.where(temp_df_shape==max_shape) # zwróci 0 albo 1
        temp_arr = np.zeros((max_shape,max_shape,3)) ## mozna czymś wypełnić tło

        if max_shape_orient[0][0] == 0:
            t = (temp_df_shape[0] - temp_df_shape[1])//2
            temp_arr[:,t:t+temp_df_shape[1],:] = temp_df

        elif max_shape_orient[0][0] == 1:
            t = (temp_df_shape[1] - temp_df_shape[0])//2
            temp_arr[t:t+temp_df_shape[0],:,:] = temp_df
        

        data_x[i] = resize(temp_arr, (256, 256))

    data_x = np.where(data_x==0,mean/i,data_x)
            


    return data_x, data_y