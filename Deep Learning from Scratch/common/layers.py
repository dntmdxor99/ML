from multiprocessing import pool
from typing import Collection
import numpy as np
from Activation_Function.Softmax_Function import softmax
from Loss_Function.cross_entropy_error import cross_entropy
from utils import im2col, col2im


class Relu:
    def __init__(self) -> None:
        self.mask = None

    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out


    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out = None

    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

 
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out


    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.x = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)

        return self.loss


    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0) -> None:
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad


    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, stride = self.stride, pad = self.pad)
        col_W = self.W.reshape(FN , -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0) -> None:
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad


    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, stride = self.stride, pad = self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(axis = 1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

        
    """
    def backward(self, dout):       # dout = (1, 1, 1, 1)
        dout = dout.transpose(0, 2, 3, 1)       # dout = (1, 1, 1, 1)
        
        pool_size = self.pool_h * self.pool_w       # (1, 2, 2) max pooling 가정, pool_size = 4
        dmax = np.zeros((dout.size, pool_size))     # dmax = (1, 4), [[0., 0., 0., 0.,]]
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()     # 
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx"""


if __name__ == "__main__":
    pass