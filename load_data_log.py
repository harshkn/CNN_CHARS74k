import sys
import os
import numpy as np

import glob
import os
from scipy import misc


def load_dataset():    
   
    path = os.getcwd()
    def load_images(dirname, s_range, e_range):
        # files = os.listdir(dirname)
        files = []
        for name in range(s_range,e_range):
            # print(name)
            files.append(str(name) + '.bmp')

        # print(files[10])
        data = np.array([misc.imread(dirname + '/'+ file, flatten=1) for file in files], dtype=np.float64)
        # print(data.shape)
        data = np.expand_dims(data, axis=1)
        # print(type(data))
        # data = np.asmatrix(data)
        # print(type(data))
        # print(data.shape)
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
        # print(arr.shape)
        return new_i

    

    X_train = load_images("trainResized32", 1, 6284)
    y_train = load_labels('trainLabels.csv')
    X_test = load_images("testResized32", 6284, 12504)

    print(X_train.shape)
    print(len(y_train))
    # X_train /= X_train.std(axis = None)
    # X_train -= X_train.mean()

    # X_test /= X_test.std(axis = None)
    # X_test -= X_test.mean()
    X_val = []
    y_val = []



# arr = []
# for i in y_train['Class']:
#     if i not in arr:
#         arr.append(i)
# print(type(arr))


# new_arr = getClassNumbers(arr)
# for idx, arr_ in enumerate(arr):   
#     print(arr_ ,new_arr[idx]) 
# [print([str(arr_), dictionary[ind]]) for ind,arr_ in enumerate(arr)]

    image_height = 32
    image_width = 32
        
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[2] * X_train.shape[3]))
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[2] * X_test.shape[3]))

    # X_train /= X_train.std(axis = None)
    # X_train -= X_train.mean()

    # X_test /= X_test.std(axis = None)
    # X_test -= X_test.mean()

    X_train, X_val = X_train[:-500], X_train[-500:]
    y_train, y_val = y_train[:-500], y_train[-500:]
    # name2 = name1[-500:]
    # X_train = X_train.reshape(X_train.shape[0], 1, 
    #               image_height, image_width).astype('float32')
   
    # X_test = X_test.reshape(X_test.shape[0], 1, 
    #               image_height, image_width).astype('float32')


    return X_train, y_train, X_val, y_val,X_test
