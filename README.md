# MatConvNet-based implementation of Fully-Convolutional Networks for image segmentation

> This code is still in alpha release. While it can be used to train
> good models or fine-tune them, we are still tuning the training to
> match the performance of the original FCN models.

This package contains an implementation of the FCN models (training
and evaluation) using the MatConvNet library.

For training, look at the `fcnTrain.m` script, and for evaluation at
`fcnTest.m`. The script `fcnTestModelZoo.m` is designed to test third
party networks imported in MatConvNet (mainly from Caffe).
