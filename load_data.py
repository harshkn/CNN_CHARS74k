import sys
import os
import numpy as np

import glob
import os
from scipy import misc


def load_dataset():    
   
    path = os.getcwd()
    def load_images(dirname):
        files = os.listdir(dirname)
        data = np.array([misc.imread(dirname + '/'+ file, flatten=1) for file in os.listdir(dirname)], dtype=np.float64)
        # print(data.shape)
        data = np.expand_dims(data, axis=1)
        # print(type(data))
        # data = np.asmatrix(data)
        # print(type(data))

        return data


    def load_labels(filename):
        data = np.genfromtxt(filename, dtype=None, delimiter=',', names=True)
        # print(data)
        # print(type(data))
        new_i = []
        arr = []
        for i in data['Class']:
            new_i.append(i)
        for j in data['Class']:
            if j not in arr:
                arr.append(j)
        print('Classes ',len(arr))

        return new_i

    

    X_train = load_images("trainResized")
    y_train = load_labels('trainLabels.csv')
    X_train, X_val = X_train[:-1000], X_train[-1000:]
    y_train, y_val = y_train[:-1000], y_train[-1000:]
    X_test = load_images("testResized")



# arr = []
# for i in y_train['Class']:
#     if i not in arr:
#         arr.append(i)
# print(type(arr))


# new_arr = getClassNumbers(arr)
# for idx, arr_ in enumerate(arr):   
#     print(arr_ ,new_arr[idx]) 
# [print([str(arr_), dictionary[ind]]) for ind,arr_ in enumerate(arr)]


        
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[2] * X_train.shape[3]))
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[2] * X_test.shape[3]))
    # X_val = X_val.reshape((X_val.shape[0], X_val.shape[2] * X_val.shape[3]))


    return X_train, y_train, X_val, y_val, X_test
