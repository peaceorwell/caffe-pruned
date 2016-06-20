#!/usr/bin/env sh

time  ./build/tools/caffe test  -model  fix_python/lenet_train_test.prototxt  -weights  fix_python/lenet_iter_10000.caffemodel    -iterations 100 -gpu all  
