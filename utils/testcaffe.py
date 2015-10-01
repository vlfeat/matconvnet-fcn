#!/usr/bin/python
# make -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-6.5  .

import os
import scipy
import scipy
import scipy.io

from skimage import io
from skimage import img_as_float
import numpy as np
import matplotlib.pyplot as plt
import sys

#model = '32s'
model = '16s'
#model = '8s'

rgb = [122.67891434, 116.66876762, 104.00698793]

filename = '/home/vedaldi/src/deep-seg/data/voc11/JPEGImages/2008_000073.jpg'

# read and convert an input image
im = io.imread(filename)
im = img_as_float(im) * 255 ;
im = np.array(im)
im = im - np.array(rgb).reshape([1, 1, 3])
im = im[:,:, ::-1] # RGB -> BGR
im = np.transpose(im,[2,0,1])
im = im.reshape([1] + list(im.shape))

# load Caffe and network
caffe_root = '/home/vedaldi/src/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

if model == '32s':
    net = caffe.Classifier('fcn-32s-pascal-deploy.prototxt',
                           'fcn-32s-pascal.caffemodel')
elif model == '16s':
    net = caffe.Classifier('fcn-16s-pascal-deploy.prototxt',
                           'fcn-16s-pascal.caffemodel')

elif model == '8s':
    net = caffe.Classifier('fcn-8s-pascal-deploy.prototxt',
                           'fcn-8s-pascal.caffemodel')
else:
    print 'Unknown model', model
    sys.exit(1)

dir(caffe.Classifier)

net.blobs['data'].reshape(*(im.shape))
net.blobs['data'].data[...] = im
net.forward()

blobs = {}
for name in net.blobs.keys():
    ename = name.replace('-', '_')
    blobs[ename] = np.array(net.blobs[name].data)

scipy.io.savemat('blobs' + model + '.mat', blobs, oned_as='column')
