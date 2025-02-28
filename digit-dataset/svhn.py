import numpy
import os
import sys

from scipy.io import loadmat

import util
from urllib.parse import urljoin
import gzip
import struct
import operator
import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class SVHN:
    base_url = 'http://ufldl.stanford.edu/housenumbers/'
    data_files = {
        'train': 'train_32x32.mat',
        'test': 'test_32x32.mat',
        # 'extra': 'extra_32x32.mat',
    }

    def __init__(self, path=None, select=[], shuffle=True, output_size=[32, 32, 3], split='train'):
        self.image_shape = (32, 32, 3)
        self.label_shape = ()
        self.path = path
        self.shuffle = shuffle
        self.output_size = output_size
        self.split = split
        self.select = select
        self.download()
        self.pointer = 0
        self.load_dataset()
        self.classpaths = []
        self.class_pointer = 10 * [0]
        for i in range(10):
            self.classpaths.append([])
        for j in range(len(self.labels)):
            label = self.labels[j]
            self.classpaths[label].append(j)

    def download(self):
        data_dir = self.path
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in list(self.data_files.values()):
            path = self.path + '/' + filename
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def shuffle_data(self):
        images = self.images[:]
        labels = self.labels[:]
        self.images = []
        self.labels = []

        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def load_dataset(self):
        abspaths = {name: self.path + '/' + path
                    for name, path in list(self.data_files.items())}

        if self.split == 'train':
            train_mat = loadmat(abspaths['train'])
            train_images = train_mat['X'].transpose((3, 0, 1, 2))
            train_labels = train_mat['y'].squeeze()
            train_labels[train_labels == 10] = 0
            train_images = train_images.astype(np.float32) / 255.
            self.images = train_images
            self.labels = train_labels
        elif self.split == 'test':
            test_mat = loadmat(abspaths['test'])
            test_images = test_mat['X'].transpose((3, 0, 1, 2))
            test_images = test_images.astype(np.float32) / 255.
            test_labels = test_mat['y'].squeeze()
            test_labels[test_labels == 10] = 0
            self.images = test_images
            self.labels = test_labels
        if len(self.select) != 0:
            self.images = self.images[self.select]
            self.labels = self.labels[self.select]

    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()

    def reset_class_pointer(self, i):
        self.class_pointer[i] = 0
        if self.shuffle:
            self.classpaths[i] = np.random.permutation(self.classpaths[i])

    def class_next_batch(self, num_per_class):
        batch_size = 10 * num_per_class
        selfimages = np.zeros((0, 32, 32, 3))
        selflabels = []
        for i in range(10):
            selfimages = np.concatenate((selfimages, self.images[
                self.classpaths[i][self.class_pointer[i]:self.class_pointer[i] + num_per_class]]), 0)
            selflabels += self.labels[self.classpaths[i][self.class_pointer[i]:self.class_pointer[i] + num_per_class]]
            self.class_pointer[i] += num_per_class
            if self.class_pointer[i] + num_per_class >= len(self.classpaths[i]):
                self.reset_class_pointer(i)
        return self.reshapes(np.array(selfimages)), get_one_hot(selflabels, 10)

    def next_batch(self, batch_size):
        images = self.images[self.pointer:(self.pointer + batch_size)]
        labels = self.labels[self.pointer:(self.pointer + batch_size)]
        self.pointer += batch_size
        if self.pointer + batch_size >= len(self.labels):
            self.reset_pointer()
        return self.reshapes(np.array(images)), get_one_hot(labels, 10)

    def reshapes(self, images):
        if images.shape[1:] == tuple(self.output_size):
            return images
        else:
            if images.shape[-1] == 1 and self.output_size[-1] == 3:
                images = gray2rgb(images)
            elif images.shape[-1] == 3 and self.output_size[-1] == 1:
                images = rgb2gray(images)

            if images.shape[1:] != tuple(self.output_size):
                images = np.array([resize(image, self.output_size) for image in images])
            return images


def main():
    svhn = SVHN(path='data/svhn')
    a, b = svhn.class_next_batch(1)
    print(a)
    print(b)


if __name__ == '__main__':
    main()
