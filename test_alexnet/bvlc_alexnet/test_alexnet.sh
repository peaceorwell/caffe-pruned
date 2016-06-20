#!/usr/bin/env sh


/home/wangd/caffe/build/tools/caffe  test -model  test_alexnet/bvlc_alexnet/train_val.prototxt  -weights  test_alexnet/bvlc_alexnet/alexnet_original_iter_4000.caffemodel    -iterations 100 -gpu all
 
