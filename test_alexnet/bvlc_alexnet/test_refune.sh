#!/usr/bin/env sh

 ./build/tools/caffe  test -model  test_alexnet/bvlc_alexnet/train_val.prototxt  -weights  test_alexnet/bvlc_alexnet/fixed.caffemodel    -iterations 100 -gpu all
