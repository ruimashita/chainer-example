#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys

import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.links import caffe
from chainer.serializers import load_npz

parser = argparse.ArgumentParser(
    description='Evaluate a Caffe reference model on ILSVRC2012 dataset')
parser.add_argument('model_type', choices=('alexnet', 'caffenet', 'googlenet'),
                    help='Model type (alexnet, caffenet, googlenet)')
parser.add_argument('--basepath', '-b', default='/',
                    help='Base path for images in the dataset')
parser.add_argument('--mean', '-m', default='data/ilsvrc_2012_mean.npy',
                    help='Path to the mean file')
parser.add_argument('--batchsize', '-B', type=int, default=100,
                    help='Minibatch size')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='Zero-origin GPU ID (nevative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np
assert args.batchsize > 0


dataset = [
    ('data/fox.jpg', 0),
    ('data/BIRD_PARK_8_0189.jpg', 0),
    ('data/fox.jpg', 0),
]
from caffe import CaffeNet
assert len(dataset) % args.batchsize == 0
model = CaffeNet()
print(model)
# func = load_npz('data/test.npz', model)
# print(func)

import pickle
dump_file = 'data/bvlc_reference_caffenet.caffemodel.pkl'
f = file(dump_file, 'r')
func = pickle.load(f)
print(func)
# print(func.fs)


    
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    func.to_gpu()

in_size = 227
mean_image = np.load(args.mean)

synset_words = np.array([line[:-1] for line in open('data/synset_words.txt', 'r')])
print(synset_words.shape)

def forward(x, t):
    y, = func(inputs={'data': x}, outputs=['fc8'], train=True)
    # print(y.data)
    indexer = y.data.argmax(axis=1)
    print(indexer)
    print(indexer[0])
    index = int(indexer[0])
    print(index)
    print(index.__class__.__name__)
    print(y.data.shape)
    print(y.data[0, 0])
    print(synset_words[index])
    print(y.data[0, index])
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size
mean_image = mean_image[:, start:stop, start:stop].copy()

x_batch = np.ndarray((args.batchsize, 3, in_size, in_size), dtype=np.float32)
y_batch = np.ndarray((args.batchsize,), dtype=np.int32)

i = 0
count = 0
accum_loss = 0
accum_accuracy = 0
for path, label in dataset:
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)[::-1]
    image = image[:, start:stop, start:stop].astype(np.float32)
    image -= mean_image

    x_batch[i] = image
    y_batch[i] = label
    i += 1

    if i == args.batchsize:
        x_data = xp.asarray(x_batch)
        y_data = xp.asarray(y_batch)

        x = chainer.Variable(x_data, volatile=True)
        t = chainer.Variable(y_data, volatile=True)

        loss, accuracy = forward(x, t)

        import chainer.computational_graph as c
        with open('data/graph.dot', 'w') as o:
            o.write(c.build_computational_graph((loss,)).dump())

        accum_loss += float(loss.data) * args.batchsize
        accum_accuracy += float(accuracy.data) * args.batchsize
        del x, t, loss, accuracy

        count += args.batchsize
        print('{} / {}'.format(count, len(dataset)), end='\r', file=sys.stderr)
        sys.stderr.flush()

        i = 0


print('mean loss:     {}'.format(accum_loss / count))
print('mean accuracy: {}'.format(accum_accuracy / count))
