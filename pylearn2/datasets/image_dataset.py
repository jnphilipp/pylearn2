__author__ = 'jnphilipp'
__license__ = 'GPL'

from PIL import Image
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
import numpy as np
import os

class ImageDataset(DenseDesignMatrix):
    def __init__(self, name, which_set, has_header=False,
                    delimiter=',', image_format='png', image_converter='RGB'):

        if which_set not in ['train', 'test', 'valid']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test", "valid"].')

        data_path = serial.preprocess('${PYLEARN2_DATA_PATH}')
        image_path = os.path.join(data_path, name, which_set)
        label_file = os.path.join(data_path, name, 'labels.csv')
        ylabel_file = os.path.join(data_path, name, 'ylabels.csv')

        labels = self.load_labels(label_file, has_header, delimiter)
        number_labels = len(set(labels.values()))

        if os.path.exists(ylabels_path):
            ylabels = {int(v):k for k,v in np.loadtxt(ylabels_path,
                                            delimiter=',',
                                            dtype=str,
                                            usecols=(0,1))}
            a == set(labels.values())
            b == set(ylabels.values())
            if not a >= b:
                ylabels = self.get_ylabels(labels, ylabels_path)
        else:
            ylabels = self.get_ylabels(labels, ylabels_path)

        imgs = [img for img in os.listdir(image_path)
                    if img.endswith(image_format)]

        img = np.array(Image.open(os.path.join(image_path,
                        imgs[0])).convert(image_converter))
        data = np.zeros(shape=(len(imgs),
            img.shape[0],
            img.shape[1],
            img.shape[2] if len(img.shape) == 3 else 1))
        y = np.zeros(shape=(len(imgs), number_labels))
        for i in range(0, len(imgs)):
            img = np.array(Image.open(os.path.join(image_path, imgs[i]))
                    .convert(image_converter))
            data[i] = img.reshape(img.shape[0],
                                    img.shape[1],
                                    img.shape[2] if len(img.shape) == 3 else 1)

            y[i][ylabels[labels[imgs[i]]]] = 1
        super(ImageDataset, self).__init__(topo_view=data, y=y)

    def load_labels(self, label_file, has_header=False, delimiter=','):
        return {k:v for k,v in np.loadtxt(label_file,
                                            delimiter=delimiter,
                                            skiprows=1 if has_header else 0,
                                            dtype=str,
                                            usecols=(0,1))}

    def get_ylabels(self, labels, ylabel_file):
        ylabels = {v:i for i,v in enumerate(set(labels.values()))}
        np.savetxt(ylabel_file, np.array(ylabels.items()), fmt=('%s', '%s'),
            delimiter=',')
        return ylabels