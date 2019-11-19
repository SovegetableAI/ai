import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import pathlib

from sklearn.datasets import load_iris
from NeuralNet import *

fn = "iris.npz"

def pretreatment():
    x,y = load_iris(True)
    y=np.reshape(y,(150,1))
    np.savez('iris.npz',data=x,label=y)


if __name__ == '__main__':
    path = pathlib.Path('iris.npz')
    if path.exists():
        pass
    else:
        pretreatment()
    nc = 3
    reader = DataReader(fn)
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(nc, base=1)


    ni = 4
    params = HyperParameters(ni, nc, eta=0.15, max_epoch=150, batch_size=5, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet(params)
    net.train(reader, checkpoint=1)


    pass
