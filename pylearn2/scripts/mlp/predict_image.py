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

import argparse
import csv
import numpy as np
import os
import sys

from PIL import Image
from pylearn2.utils import serial
from theano import function
from theano import tensor as T


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from sys.argv
    """
    parser = argparse.ArgumentParser(
        description='Launch a prediction from a pkl file'
    )
    parser.add_argument('model_filename',
                        help='Specifies the pkl model file')
    parser.add_argument('name',
                        help='Name of the dataset')
    parser.add_argument('which_set',
                        help='which set to use (train/test/valid)')
    parser.add_argument('output_filename',
                        nargs='?',
                        help='Specifies the predictions output file')
    parser.add_argument('--prediction_type', '-P',
                        default="classification",
                        help='Prediction type (classification/regression)')
    parser.add_argument('--image_format', '-F',
                        default='png',
                        help='File extension of the images')
    parser.add_argument('--convert_mode', '-C',
                        default='RGB',
                        help='Convert mode')
    return parser

def predict(model_path, name, which_set, output_path,
            predictionType='classification', image_format='png',
            convert_mode='RGB'):
    """
    Predict from a pkl file.

    Parameters
    ----------
    model_path : str
        The file name of the model file.
    name : str
        The file name of the file or folder to test/predict.
    which_set : str
        The file name of the file or folder to test/predict.
    output_path : str
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

    print('loading data...')
    data_path = serial.preprocess('${PYLEARN2_DATA_PATH}')
    image_path = os.path.join(data_path, name, which_set)
    imgs = [img for img in os.listdir(image_path)
                if img.endswith(image_format)]

    img = np.array(Image.open(os.path.join(image_path, imgs[0]))
                    .convert(convert_mode))
    x = np.zeros(shape=(len(imgs),
                        img.shape[0],
                        img.shape[1],
                        img.shape[2] if len(img.shape) == 3 else 1))

    for i in range(0, len(imgs)):
        img = np.array(Image.open(os.path.join(image_path, imgs[i]))
                        .convert(convert_mode))
        x[i] = img.reshape(img.shape[0],
                            img.shape[1],
                            img.shape[2] if len(img.shape) == 3 else 1)

    # predict iamges
    print('predicting images...')
    y = f(x)

    print('writing predictions...')
    classes = {}
    with open(os.path.join(data_path, name, 'classes.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes[row['image']] = int(row['class'])

    predictions = []
    tp = 0
    for i in range(0, len(imgs)):
        print('%s: %s%s' % (imgs[i],
                            y[i],
                            ' (%s)' % classes[imgs[i]]))
        if y[i] == classes[imgs[i]]: tp += 1
        predictions.append({'image':imgs[i], 'predicted':y[i], 'class':classes[imgs[i]]})
    print('precision: %.4f' % (tp / len(imgs)))

    if output_path:
        with open(output_path, 'w') as f:
            writer = csv.DictWriter(f, ['image', 'predicted', 'class'], dialect='unix')
            writer.writeheader()
            for prediction in predictions:
                writer.writerow(prediction)

    return True

if __name__ == "__main__":
    """
    See module-level docstring for a description of the script.
    """
    parser = make_argument_parser()
    args = parser.parse_args()
    ret = predict(args.model_filename, args.name, args.which_set,
                    args.output_filename, args.prediction_type,
                    args.image_format, args.convert_mode)
    if not ret:
        sys.exit(-1)
