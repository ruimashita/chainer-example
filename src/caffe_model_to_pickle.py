#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chainer.functions.caffe import CaffeFunction
import pickle

print('Load model')

func = CaffeFunction('data/bvlc_reference_caffenet.caffemodel')

pickle.dump(func, open('data/bvlc_reference_caffenet.caffemodel.pkl', 'wb'))

print('dumped')
