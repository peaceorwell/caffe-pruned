#!/usr/bin/env sh
 
./build/tools/caffe  test   -model  fix_python/lenet_train_test.prototxt  -weights  fix_python/refune_iter_5000.caffemodel    -iterations 1000 -gpu all -step three 
