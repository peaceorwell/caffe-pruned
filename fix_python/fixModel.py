import types
from compiler.ast import flatten
import numpy as np
import matplotlib.pyplot as plt
caffe_root='../'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
model_def=caffe_root+'examples/mnist/deploy.prototxt'
model_caffe=caffe_root+'fix_python/lenet_iter_5000.caffemodel'
net=caffe.Net(model_def,model_caffe,caffe.TEST)

def getVpt(v,k):
    v=abs(v)
    vList=v.tolist()
    vList=flatten(vList)
    vList.sort()
    k=(int)(k*len(vList))
    return vList[k]



for k,v in net.params.items():
    idx=v[0].data.shape
    print (k)
#    print (type(v[0].data))
    cnt=0
    count=0
    vpt=getVpt(v[0].data,0.95)
    if len(idx)==2:
        for h_idx in range(0,idx[0]):
            for w_idx in range(0,idx[1]):
                count=count+1
                if v[0].data[h_idx][w_idx]< vpt and v[0].data[h_idx][w_idx]>-1*vpt:
                    cnt=cnt+1
                    v[0].data[h_idx][w_idx]=0.0
                    v[0].mask[h_idx][w_idx]=0.0
        print (cnt)
        print (count)
    print (v[0].data)
net.save('fixed_0.95.caffemodel')


