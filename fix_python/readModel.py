import types
from compiler.ast import flatten
import numpy as np
import matplotlib.pyplot as plt
caffe_root='../'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
model_def=caffe_root+'examples/mnist/deploy.prototxt'
model_caffe=caffe_root+'fix_python/refune_iter_10000.caffemodel'
net=caffe.Net(model_def,model_caffe,caffe.TEST)

for k,v in net.params.items():
    print k
    if k == 'ip2':
#        print v[0].mutable_shape
#        m=v[0].data
        print v[0].count
        print v[0].nnz
#        print v[0].num_axes
#        print v[0].csrrowptr
        print v[0].csrval


