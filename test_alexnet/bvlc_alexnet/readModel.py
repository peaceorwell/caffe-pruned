import types
from compiler.ast import flatten
import numpy as np
import matplotlib.pyplot as plt
caffe_root='../../'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
model_def=caffe_root+'test_alexnet/bvlc_alexnet/deploy.prototxt'
#model_caffe=caffe_root+'test_alexnet/bvlc_alexnet/alexnet_train_iter_4000.caffemodel'
model_caffe=caffe_root+'test_alexnet/bvlc_alexnet/fixed.caffemodel'
net=caffe.Net(model_def,model_caffe,caffe.TEST)

def getVpt(v,k):
    v=abs(v)
    vList=v.tolist()
    vList=flatten(vList)
    vList.sort()
    k=(int)(k*len(vList))
    return vList[k]



for k,v in net.params.items():
    if k=='fc7':
        print v[0].count
        print v[0].diff
        print v[0].mask


