from HelperClass.NeuralNet_1_1 import *

# main
if __name__ == '__main__':
    # data
    reader = DataReader_1_1('housing.npz')
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    hp = HyperParameters_1_0(13, 1, eta=0.01, max_epoch=3000, batch_size=20, eps=1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    print("W=", net.W)
    print("B=", net.B)
    # inference
    x1 = 0.00632
    x2 = 18.00

    x3 = 2.310
    x4 = 0
    x5 = 0.5380
    x6 = 6.5750
    x7 = 65.20
    x8 = 4.0900
    x9 = 1
    x10 = 296.0
    x11 = 15.30
    x12 = 396.90
    x13 = 4.98
    x = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]).reshape(1,13)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    print("z=", z)
    Z_true = z * reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print("Z_true=", Z_true)
