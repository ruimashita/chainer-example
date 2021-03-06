#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chainer.functions.caffe import CaffeFunction
from chainer.serializers import save_npz

print('Load model')

func = CaffeFunction('data/bvlc_reference_caffenet.caffemodel')

save_npz('data/bvlc_reference_caffenet.caffemodel.npz', func)

print('Loaded')
