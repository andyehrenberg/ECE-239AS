# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

from nndl.layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *


class MyConvNet2(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 32], filter_size=3,
                 hidden_dims=[100, 100], num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: List of size  Nbconv+1 with the number of filters
        to use in each convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dims: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        self.filter_size = filter_size

        Cinput, Hinput, Winput = input_dim
        stride_conv = 1

        F = [Cinput] + num_filters
        for i in range(len(num_filters)):
            idx = i + 1
            W = weight_scale * \
                np.random.randn(F[i + 1], F[i], filter_size, filter_size)
            b = np.zeros(F[i + 1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(F[i + 1]),
                            'running_var': np.zeros(F[i + 1])}
                gamma = np.ones(F[i + 1])
                beta = np.zeros(F[i + 1])
                self.bn_params.update({
                    'bn_param' + str(idx): bn_param})
                self.params.update({
                    'gamma' + str(idx): gamma,
                    'beta' + str(idx): beta})

        Hconv, Wconv = self.Size_Conv(
            stride_conv, filter_size, Hinput, Winput, len(num_filters))
        dims = [Hconv * Wconv * F[-1]] + hidden_dims
        for i in range(len(hidden_dims)):
            idx = len(num_filters) + i + 1
            W = weight_scale * np.random.randn(int(dims[i]), int(dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(dims[i + 1]),
                            'running_var': np.zeros(dims[i + 1])}
                gamma = np.ones(int(dims[i + 1]))
                beta = np.zeros(int(dims[i + 1]))
                self.bn_params.update({
                    'bn_param' + str(idx): bn_param})
                self.params.update({
                    'gamma' + str(idx): gamma,
                    'beta' + str(idx): beta})

        W = weight_scale * np.random.randn(dims[-1], num_classes)
        b = np.zeros(num_classes)
        self.params.update({'W' + str(len(num_filters) + len(hidden_dims) + 1): W,
                            'b' + str(len(num_filters) + len(hidden_dims) + 1): b})

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def Size_Conv(self, stride_conv, filter_size, H, W, Nbconv):
        P = (filter_size - 1) / 2
        Hc = (H + 2 * P - filter_size) / stride_conv + 1
        Wc = (W + 2 * P - filter_size) / stride_conv + 1
        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1
        if Nbconv == 1:
            return Hp, Wp
        else:
            H = Hp
            W = Wp
            return self.Size_Conv(stride_conv, filter_size, H, W, Nbconv - 1)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        N = X.shape[0]

        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.items():
                bn_param[mode] = mode

        scores = None
       
        #Forward Pass
        
        blocks = {}
        blocks['h0'] = X

        for i in range(self.L):
            idx = i + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = blocks['h' + str(idx - 1)]
            if self.use_batchnorm:
                beta = self.params['beta' + str(idx)]
                gamma = self.params['gamma' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]
                h, cache_h = conv_norm_relu_pool_forward(
                    h, w, b, conv_param, pool_param, gamma, beta, bn_param)
            else:
                h, cache_h = conv_relu_pool_forward(
                    h, w, b, conv_param, pool_param)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        for i in range(len(hidden_dims)):
            idx = len(num_filters) + i + 1
            h = blocks['h' + str(idx - 1)]
            if i == 0:
                h = h.reshape(N, np.product(h.shape[1:]))
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            if self.use_batchnorm:
                beta = self.params['beta' + str(idx)]
                gamma = self.params['gamma' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]
                h, cache_h = affine_norm_relu_forward(h, w, b, gamma,
                                                      beta, bn_param)
            else:
                h, cache_h = affine_relu_forward(h, w, b)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        idx = len(num_filters) + len(hidden_dims) + 1
        w = self.params['W' + str(idx)]
        b = self.params['b' + str(idx)]
        h = blocks['h' + str(idx - 1)]
        h, cache_h = affine_forward(h, w, b)
        blocks['h' + str(idx)] = h
        blocks['cache_h' + str(idx)] = cache_h

        scores = blocks['h' + str(idx)]

        if y is None:
            return scores

        loss, grads = 0, {}
        
        #Calculate Loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss = data_loss + reg_loss
        
        #Backwards Pass
        idx = len(num_filters) + len(hidden_dims) + 1
        dh = dscores
        h_cache = blocks['cache_h' + str(idx)]
        dh, dw, db = affine_backward(dh, h_cache)
        blocks['dh' + str(idx - 1)] = dh
        blocks['dW' + str(idx)] = dw
        blocks['db' + str(idx)] = db

        for i in range(len(hidden_dims))[::-1]:
            idx = len(num_filters) + i + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(
                    dh, h_cache)
                blocks['dbeta' + str(idx)] = dbeta
                blocks['dgamma' + str(idx)] = dgamma
            else:
                dh, dw, db = affine_relu_backward(dh, h_cache)
            blocks['dh' + str(idx - 1)] = dh
            blocks['dW' + str(idx)] = dw
            blocks['db' + str(idx)] = db

        for i in range(len(num_filters))[::-1]:
            idx = i + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if i == max(range(len(num_filters))[::-1]):
                dh = dh.reshape(*blocks['h' + str(idx)].shape)
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = conv_norm_relu_pool_backward(
                    dh, h_cache)
                blocks['dbeta' + str(idx)] = dbeta
                blocks['dgamma' + str(idx)] = dgamma
            else:
                dh, dw, db = conv_relu_pool_backward(dh, h_cache)
            blocks['dh' + str(idx - 1)] = dh
            blocks['dW' + str(idx)] = dw
            blocks['db' + str(idx)] = db

        list_dw = {key[1:]: val + self.reg * self.params[key[1:]]
                   for key, val in blocks.items() if key[:2] == 'dW'}
        list_db = {key[1:]: val for key, val in blocks.items() if key[:2] ==
                   'db'}
        list_dgamma = {key[1:]: val for key, val in blocks.items() if key[
            :6] == 'dgamma'}
        list_dbeta = {key[1:]: val for key, val in blocks.items() if key[
            :5] == 'dbeta'}

        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)

        return loss, grads