#!/usr/bin/env sh

./build/tools/caffe  train -solver=fix_python/lenet_fixed_solver.prototxt  -weights  fix_python/fixed_0.95.caffemodel -gpu all  -step two
