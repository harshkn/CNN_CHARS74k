import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.io as sio
import os 
import random
#from sklearn.metrics import accuracy_score
# from sklearn import datasets, svm, metrics
from sklearn import neighbors, datasets ,metrics   

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from load_data_log import load_dataset
# sys.stdout= open("output_logreg.txt","w")

#Download the dataset 

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)



# t_sample_sizes = [100, 1000, 5000, 10000, 20000, 40000, 55000]
# got 38 % accuracy on kaggle

def getClassNumbers(an_class):
        # dictionary = ['1','2','3','4','5','6','7','8','9','0','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        # 'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
        # 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        n_class = []
        dictionary_ = {'1':'0','2':'1','3':'2','4':'3','5':'4','6':'5','7':'6','8':'7','9':'8','0':'9','a':'10', 'b':'11', 'c':'12', 'd':'13', 'e':'14', 'f':'15', 'g':'16', 'h':'17', 'i':'18', 'j':'19', 'k':'20', 'l':'21', 
        'm':'22', 'n':'23', 'o':'24', 'p':'25', 'q':'26','r':'27', 's':'28', 't':'29', 'u':'30', 'v':'31', 'w':'32', 'x':'33', 'y':'34', 'z':'35','A':'36', 'B':'37', 'C':'38', 'D':'39', 'E':'40', 'F':'41', 'G':'42', 'H':'43', 'I':'44', 'J':'45', 'K':'46', 'L':'47', 
        'M':'48', 'N':'49', 'O':'50', 'P':'51', 'Q':'52','R':'53', 'S':'54', 'T':'55', 'U':'56', 'V':'57', 'W':'58', 'X':'59', 'Y':'60', 'Z':'61'}

        for each_entry in an_class:
            n_class.append(dictionary_[each_entry.decode("utf-8")])
        # print(type(n_class))
        n_class = np.asarray(n_class,dtype=np.uint8)
        # print(type(n_class))
        return n_class

image_height = 32
image_width = 32
classes = 62

X_train, y_train, X_val, y_val,X_test= load_dataset()
y_train = getClassNumbers(y_train)
y_val = getClassNumbers(y_val)


# print(X_train[0].shape)
# plt.imshow(X_train[0])
# plt.show()

X_train = X_train.reshape(X_train.shape[0], 
                  image_height * image_width).astype('float32')

X_test = X_test.reshape(X_test.shape[0],  
              image_height * image_width).astype('float32')

print('Total size of training labels is ', y_train.shape)
print('Total size of training images is ', X_train.shape)
print('Total size of test images is ', X_test.shape)

t_sample_sizes = [y_train.shape[0]]
# t_sample_sizes = [1000]

acc = [];
for idx, t_sample_size in enumerate(t_sample_sizes):

	#train a logistic regression based classifier from scikitlearn
	logreg =  neighbors.KNeighborsClassifier()
	r_index = random.sample(range(len(y_train)),t_sample_size)
	logreg.fit(X_train[r_index,:], y_train[r_index]) 
	# logreg.fit(X_train, y_train) 

	res = logreg.predict(X_test)
	# accuracy = metrics.accuracy_score(y_val, res)
	# acc.append(metrics.accuracy_score(y_val, res))
	# images = logreg.coef_.reshape((classes ,image_height, image_width))
	# print(logreg.coef_.shape)
	# print(res)
	# print(res.shape)

	# image = images[0,:,:];
	# for i in range(1,10):
	# 	image = np.c_[image,images[i,:,:]]
		
	
	print('Number of training samples : ', t_sample_size)
	# print('Classification report ', metrics.classification_report(mnist.test.labels, res))
	# print('Accuracy is ', accuracy)
	# print('Confusion Matrix ', metrics.confusion_matrix(y_val, res))
	print("--------------------------------------------------------------")
	print()


	
	# plt.imshow(image)
	# plt.axis('off')
	# fn = "weights_"+ str(t_sample_size) + ".png"
	# plt.savefig(fn,bbox_inches='tight')
a = []
dictionary = ['1','2','3','4','5','6','7','8','9','0','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
        'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for ind, res_ in enumerate(res):
    # print([ind , dictionary[res_]])
    a.append([ind+6284, dictionary[res_]])

np.savetxt("submission_log_knn.csv", a, delimiter=",",fmt="%s",header=('ID,Class'),comments='') 
# sys.stdout.close()

# print(acc)
# plt.plot(t_sample_sizes, acc, linewidth=2.0)
# plt.xlabel('Training sample size')
# plt.ylabel('Accuracy')
# plt.title('Accuracy with training sample size')
# plt.savefig('trvsacc.png',bbox_inches='tight')




