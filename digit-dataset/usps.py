import numpy
import os
import sys
import util
from urllib.parse import urljoin
import gzip
import struct
import operator
import numpy as np
from functools import reduce
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from scipy.io import loadmat as load


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class USPS:
    base_url = 'https://cs.nyu.edu/~roweis/data/'

    data_files = {
        'data': 'usps_all.mat', \
        }

    def __init__(self, path=None, shuffle=True, output_size=(16, 16, 1), split='train', select=[]):
        self.image_shape = (16, 16, 1)
        self.label_shape = ()
        self.path = path
        self.shuffle = shuffle
        self.output_size = output_size
        self.split = split
        self.select = select
        self.download()
        self.pointer = 0
        self.load_dataset()

    def download(self):
        data_dir = self.path
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in list(self.data_files.values()):
            path = self.path + '/' + filename
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def _extract_images(self, filename, split):
        """Extract the images into a numpy array.

        Args:
        filename: The path to an usps images file.
        num_images: The number of images in the file.

        Returns:
        A numpy array of shape [number_of_images, height, width, channels].
        A numpy array of shape [number_of_labels]
        """
        print('Extracting images from: ', filename)

        data = np.rollaxis(np.rollaxis(load(filename)['data'], 1), -1)

        if split == "train":
            data = data[:, :1000, :]
        elif split == "test":
            data = data[:, 1000:, :]

        label = np.concatenate([np.ones(data.shape[1]) * ((1 + i) % 10) for i in range(10)])
        data = np.concatenate(data).reshape((-1, 16, 16, 1)).transpose((0, 2, 1, 3))
        return data.astype(int), label.astype(int)

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
            self.images, self.labels = self._extract_images(abspaths['data'], 'train')
        elif self.split == 'test':
            self.images, self.labels = self._extract_images(abspaths['data'], 'test')
        if len(self.select) != 0:
            self.images = self.images[self.select]
            self.labels = self.labels[self.select]

    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        if self.pointer + batch_size >= len(self.labels):
            self.reset_pointer()
        images = self.images[self.pointer:(self.pointer + batch_size)]
        labels = self.labels[self.pointer:(self.pointer + batch_size)]
        self.pointer += batch_size
        return np.array(images), get_one_hot(labels, 10)

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
    mnist = MNIST(path='data/mnist')
    a, b = mnist.next_batch(1)
    print(a)
    print(b)


if __name__ == '__main__':
    main()
