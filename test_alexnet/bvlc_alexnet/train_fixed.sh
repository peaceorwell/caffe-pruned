#!/usr/bin/env sh

./build/tools/caffe  train -solver test_alexnet/bvlc_alexnet/fixed_solver.prototxt  -weights test_alexnet/bvlc_alexnet/fixed.caffemodel -gpu all  
