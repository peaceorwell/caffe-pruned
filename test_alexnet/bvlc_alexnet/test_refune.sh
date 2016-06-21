#!/usr/bin/env sh

 ./build/tools/caffe  test -model  test_alexnet/bvlc_alexnet/train_val.prototxt  -weights  test_alexnet/bvlc_alexnet/alexnet_refune_iter_4000.caffemodel    -iterations 1000 -gpu all -step three
