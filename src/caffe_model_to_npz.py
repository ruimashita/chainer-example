#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chainer.functions.caffe import CaffeFunction
from chainer.serializers import save_npz


func = CaffeFunction('data/bvlc_reference_caffenet.caffemodel')

save_npz('data/test.npz', func)

print('Loaded')
