import scipy.io as sp
import os
import numpy as np
from keras.utils import to_categorical
import cv2
import tensorflow as tf

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class Datasets():
    def __init__(self, root_path):
        self.root_path = root_path

    def __load_data__(self):
        pass

    def __split_train_test__(self):
        pass

    def get_batch(self):
        pass

class cifar10(Datasets):
    def __init__(self, root_path, test_split=0.3):
        super(cifar10, self).__init__(root_path)
        train_data = []
        train_label = []
        for i in range(1,6):
            train_path = '%s/data/cifar-10-batches-py/data_batch_%d' % (root_path, i)
            data_dic = unpickle(train_path)
            train_data.append(data_dic['data'])
            train_label = train_label + data_dic['labels']
        train_data = np.concatenate(train_data)
        test_dic = unpickle('%s/data/cifar-100-python/test' % root_path)
        test_data = test_dic['data']
        test_label = test_dic['labels']
        train_data = train_data.reshape((50000, 3, 32, 32))
        self.train_data = train_data.transpose((0,2,3,1))
        self.train_label = np.array(train_label)
        test_data = test_data.reshape((10000, 3, 32, 32))
        self.test_data = test_data.transpose((0,2,3,1))
        self.test_label = np.array(test_label)
        self.mean = None
        self.std = None

        def rotate_img(self, img, angle):
            img = tf.contrib.image.rotate(img, angle, interpolation='NEAREST')
            return img

        def set_mean_std(self, data):
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)


