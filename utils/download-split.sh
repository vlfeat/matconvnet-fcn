#!/bin/bash
# dowlnoad list of validation images for FCN paper
wget https://gist.githubusercontent.com/shelhamer/edb330760338892d511e/raw/c858e6b23cb05533c88917f933e3bab5aedddb1a/seg11valid.txt --output-document=data/seg11valid.txt

wget http://www.robots.ox.ac.uk/~szheng/Res_CRFRNN/val.txt --output-document=data/seg12valid-tvg.txt