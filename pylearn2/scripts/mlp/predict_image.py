#!/usr/bin/env python
# coding: utf-8
"""
Script to predict values using a pkl model file.

This is a configurable script to make predictions.

Basic usage:

.. code-block:: none

    predict_csv.py pkl_file.pkl test.csv output.csv

Optionally it is possible to specify if the prediction is regression or
classification (default is classification). The predicted variables are
integer by default.
Based on this script: http://fastml.com/how-to-get-predictions-from-pylearn2/.
This script doesn't use batches. If you run out of memory it could be 
resolved by implementing a batch version.

"""

__author__ = 'jnphilipp'
__license__ = 'GPL'

import sys
import os
import argparse
import numpy as np

from PIL import Image
from pylearn2.utils import serial
from theano import tensor as T
from theano import function


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from sys.argv
    """
    parser = argparse.ArgumentParser(
        description='Launch a prediction from a pkl file'
    )
    parser.add_argument('model_filename',
                        help='Specifies the pkl model file')
    parser.add_argument('test_path',
                        help='Specifies the folder or the file to predict')
    parser.add_argument('output_filename',
                        nargs='?',
                        help='Specifies the predictions output file')
    parser.add_argument('--ylabels_path', '-Y',
                        help='Specifies the path to the ylabels.csv file')
    parser.add_argument('--prediction_type', '-P',
                        default="classification",
                        help='Prediction type (classification/regression)')
    parser.add_argument('--output_type', '-T',
                        default="int",
                        help='Output variable type (int/float)')
    parser.add_argument('--image_format', '-F',
                        default='png',
                        help='File extension of the images, only neccessary if test_path is a folder')
    parser.add_argument('--convert_mode', '-C',
                        default='RGB',
                        help='Convert mode')
    return parser

def predict(model_path, test_path, output_path,
            predictionType='classification', outputType='int',
            ylabels_path=None, image_format='png', convert_mode='RGB'):
    """
    Predict from a pkl file.

    Parameters
    ----------
    model_path : str
        The file name of the model file.
    test_path : str
        The file name of the file or folder to test/predict.
    outputpath : str
        The file name of the output file.
    predictionType : str, optional
        Type of prediction (classification/regression).
    outputType : str, optional
        Type of predicted variable (int/float).
    """

    print('loading model...')

    try:
        model = serial.load(model_path)
    except Exception as e:
        print('error loading {}:'.format(model_path))
        print(e)
        return False

    print('setting up symbolic expressions...')

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    if predictionType == "classification":
        Y = T.argmax(Y, axis=1)

    f = function([X], Y, allow_input_downcast=True)

    print('loading data and predicting...')
    if os.path.isdir(test_path):
        imgs = [img for img in os.listdir(test_path)
                    if img.endswith(image_format)]

        img = np.array(Image.open(os.path.join(test_path, imgs[0]))
                        .convert(convert_mode))
        x = np.zeros(shape=(len(imgs),
                            img.shape[0],
                            img.shape[1],
                            img.shape[2] if len(img.shape) == 3 else 1))

        for i in range(0, len(imgs)):
            img = np.array(Image.open(os.path.join(test_path, imgs[i]))
                            .convert(convert_mode))
            x[i] = img.reshape(img.shape[0],
                                img.shape[1],
                                img.shape[2] if len(img.shape) == 3 else 1)
    else:
        imgs = [os.path.basename(test_path)]
        img = np.array(Image.open(test_path).convert(convert_mode))
        img = img.reshape(img.shape[0],
                        img.shape[1],
                        img.shape[2] if len(img.shape) == 3 else 1)
        x = np.zeros(shape=(1, img.shape[0], img.shape[1], img.shape[2]))
        x[0] = img

    y = f(x)

    print('writing predictions...')
    ylabels = None
    if not ylabels_path:
        ylabels_path = os.path.join(test_path, 'ylabels.csv') 
    if os.path.exists(ylabels_path):
        ylabels = {int(v):k for k,v in np.loadtxt(ylabels_path,
                                            delimiter=',',
                                            dtype=str,
                                            usecols=(0,1))}

    predictions = []
    for i in range(0, len(imgs)):
        print('%s: %s%s' % (imgs[i],
                            y[i],
                            ' (%s)' % ylabels[y[i]] if ylabels else ''))
        predictions.append([imgs[i], y[i], ylabels[y[i]] if ylabels else None])

    if output_path:
        np.savetxt(output_path, np.array(predictions),
            fmt=('%s', '%s', '%s'),
            delimiter=',')

    return True

if __name__ == "__main__":
    """
    See module-level docstring for a description of the script.
    """
    parser = make_argument_parser()
    args = parser.parse_args()
    ret = predict(args.model_filename, args.test_path, args.output_filename,
                    args.prediction_type, args.output_type, args.ylabels_path,
                    args.image_format, args.convert_mode)
    if not ret:
        sys.exit(-1)