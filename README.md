# MatConvNet implementation of the FCN models for semantic segmentation

This package contains an implementation of the FCN models (training
and evaluation) using the MatConvNet library.

For training, look at the `fcnTrain.m` script, and for evaluation at
`fcnTest.m`. The script `fcnTestModelZoo.m` is designed to test third
party networks imported in MatConvNet (mainly from Caffe).

While we are still tuning parameters, on the PASCAL VOC 2011
validation data subset used in the FCN paper, this code has been used
to train networks with this performance:

| Model           | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy |
|-----------------|-----------|---------|--------------------|----------------|
| FCN-32s (ours)  | RV-VOC11  | 60.80   | 89.61              | 75.49          |
| FCN-16s (ours)  | RV-VOC11  | 62.25   | 90.08              | 77.81          |
| FCN-8s  (ours)  | RV-VOC11  | in prog.| in prog.           | in prog.       |
| FNC-32s (orig.) | RV-VOC11  | 59.43   | 89.12              | 73.28          |
| FNC-16s (orig.) | RV-VOC11  | 62.35   | 90.02              | 75.74          |
| FNC-8s  (orig.) | RV-VOC11  | 62.69   | 90.33              | 75.86          |

The original FCN models can be downloaded from the MatConvNet
[model repository](http://www.vlfeat.org/matconvnet/pretrained/).

## About

This code was developed by

* Sebastien Ehrhardt
* Andrea Vedaldi

## References

'Fully Convolutional Models for Semantic Segmentation', *Jonathan
Long, Evan Shelhamer and Trevor Darrell*, CVPR, 2015
([paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)).

## Changes

* v0.9.1 -- Bugfixes.
* v0.9   -- Initial release. FCN32s and FCN16s work well.
