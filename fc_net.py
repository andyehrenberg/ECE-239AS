import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                dropout=0, use_batchnorm=False, reg=0.0,
                weight_scale=1e-2, dtype=np.float32, seed=None):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #
    #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
    #   parameters to zero.  The gamma and beta parameters for layer 1 should
    #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
    #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm 
    #   is true and DO NOT batch normalize the output scores.
    # ================================================================ #
        self.hidden_dims = hidden_dims
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W' + str(i+1)] = weight_scale * np.random.randn(dims[i], dims[i + 1])
            self.params['b' + str(i+1)] = np.zeros(dims[i+1])
            if self.use_batchnorm and (i != self.num_layers - 1):
                self.params['gamma' + str(i+1)] = np.ones(dims[i+1])
                self.params['beta' + str(i+1)] = np.zeros(dims[i+1])
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode   
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    #
    #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
    #   between the affine_forward and relu_forward layers.  You may
    #   also write an affine_batchnorm_relu() function in layer_utils.py.
    #
    #   DROPOUT: If dropout is non-zero, insert a dropout layer after
    #   every ReLU layer.
    # ================================================================ #
        scores = {}
        cache = {}
        dropout_cache = {}

        scores[0] = X

        if self.use_batchnorm:
            for i in range(1, self.num_layers+1):
                if i!=self.num_layers:
                    scores[i], cache[i] = affine_batchnorm_relu(scores[i-1],
                                                                self.params['W' + str(i)],
                                                                self.params['b' + str(i)],
                                                                self.params['gamma' + str(i)],
                                                                self.params['beta' + str(i)],
                                                                bn_param=self.bn_params[i-1])
                    if self.use_dropout:
                        scores[i], dropout_cache[i] = dropout_forward(
                                                      scores[i],
                                                      self.dropout_param)
    
                else:
                    scores[i], cache[i] = affine_forward(
                                          scores[i-1],
                                          self.params['W' + str(i)],
                                          self.params['b' + str(i)])
            
        else:
            for i in range(1, self.num_layers+1):
                if i!=self.num_layers:
                    scores[i], cache[i] = affine_relu_forward(
                                          scores[i-1],
                                          self.params['W' + str(i)],
                                          self.params['b' + str(i)])
                    if self.use_dropout:
                        scores[i], dropout_cache[i] = dropout_forward(
                                                      scores[i],
                                                      self.dropout_param)
                else:
                    scores[i], cache[i] = affine_forward(
                                          scores[i-1],
                                          self.params['W' + str(i)],
                                          self.params['b' + str(i)])
    
    
    
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        
        # If test mode return early
        if mode == 'test':
            return scores[self.num_layers]
    
        loss, grads = 0.0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backwards pass of the FC net and store the gradients
        #   in the grads dict, so that grads[k] is the gradient of self.params[k]
        #   Be sure your L2 regularization includes a 0.5 factor.
        #
        #   BATCHNORM: Incorporate the backward pass of the batchnorm.
        #
        #   DROPOUT: Incorporate the backward pass of dropout.
        # ================================================================ #
    
        loss, dscores = softmax_loss(scores[self.num_layers], y)
        
        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            loss += 0.5 * self.reg * np.sum(w * w)
        
        if self.use_batchnorm:
            for i in range(self.num_layers, 0, -1):
                if i!=self.num_layers:                 
                    if self.use_dropout:
                        dscores = dropout_backward(dscores, dropout_cache[i])
                    (dscores, 
                    grads['W' + str(i)], 
                    grads['b' + str(i)], 
                    grads['gamma' + str(i)], 
                    grads['beta' + str(i)]) = affine_batchnorm_relu_backward(dscores, cache[i])
                        
                    grads['W' + str(i)] += self.reg*self.params['W' + str(i)]
                else:
                    dscores, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(dscores, cache[i])
    
                    grads['W' + str(i)] += self.reg*self.params['W' + str(i)]
        
        else:
            for i in range(self.num_layers, 0, -1):
                if i!=self.num_layers:
                    if self.use_dropout:
                        dscores = dropout_backward(dscores,dropout_cache[i])
                    dscores, grads['W' + str(i)], grads['b' + str(i)] = affine_relu_backward(dscores, cache[i])
                        
                    grads['W' + str(i)] += self.reg*self.params['W' + str(i)]
                else:
                    dscores, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(dscores, cache[i])
                        
                    grads['W' + str(i)] += self.reg*self.params['W' + str(i)]
    
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        
        return loss, grads
    