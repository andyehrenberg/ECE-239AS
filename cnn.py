import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
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

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C, H, W = input_dim
    
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    H_pool = int((H-2)/2+1)
    W_pool = int((W-2)/2+1)
    
    weights_dim = [(num_filters, C, filter_size, filter_size),
                  (H_pool*W_pool*num_filters, hidden_dim),
                  (hidden_dim, num_classes)]
    biases_dim = [num_filters, hidden_dim, num_classes]
    
    for i in range(1,4):
      self.params['W%d' %i] = np.random.normal(loc=0.0, scale=weight_scale, 
                                                 size=weights_dim[i-1])
      self.params['b%d' %i] = np.zeros(biases_dim[i-1])
    
    if self.use_batchnorm:
      bn_param1 = {'mode': 'train',
                   'running_mean': np.zeros(num_filters),
                   'running_var': np.zeros(num_filters)}
      gamma1 = np.ones(num_filters)
      beta1 = np.zeros(num_filters)

      bn_param2 = {'mode': 'train',
                         'running_mean': np.zeros(hidden_dim),
                         'running_var': np.zeros(hidden_dim)}
      gamma2 = np.ones(hidden_dim)
      beta2 = np.zeros(hidden_dim)

      self.bn_params.update({'bn_param1': bn_param1,
                             'bn_param2': bn_param2})

      self.params.update({'beta1': beta1,
                          'beta2': beta2,
                          'gamma1': gamma1,
                          'gamma2': gamma2})
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    mode = 'train'
    
    if y is None:
      mode = 'test'
    
    if self.use_batchnorm:
      for key, bn_param in self.bn_params.items():
        bn_param[mode] = mode
        
    if self.use_batchnorm:
      bn_param1, gamma1, beta1 = self.bn_params['bn_param1'], self.params['gamma1'], self.params['beta1']
      bn_param2, gamma2, beta2 = self.bn_params['bn_param2'], self.params['gamma2'], self.params['beta2']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
           
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    if self.use_batchnorm:
      conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)

      batchnorm1, batchnorm1_cache = spatial_batchnorm_forward(conv_out, gamma1, beta1, bn_param1)

      conv_relu, conv_relu_cache = relu_forward(batchnorm1)

      max_out, max_cache = max_pool_forward_fast(conv_relu, pool_param)

      affine_out, affine_cache = affine_forward(max_out, W2, b2)

      batchnorm2, batchnorm2_cache = batchnorm_forward(affine_out, gamma2, beta2, bn_param2)

      affine_relu_out, affine_relu_cache = relu_forward(batchnorm2)

      scores, scores_cache = affine_forward(affine_relu_out, W3, b3)
        
    else:
      conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)

      conv_relu, conv_relu_cache = relu_forward(conv_out)

      max_out, max_cache = max_pool_forward_fast(conv_relu, pool_param)

      affine_out, affine_cache = affine_forward(max_out, W2, b2)

      affine_relu_out, affine_relu_cache = relu_forward(affine_out)

      scores, scores_cache = affine_forward(affine_relu_out, W3, b3)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    if self.use_batchnorm: 
      loss, dout = softmax_loss(scores, y)

      loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))

      daffine, grads['W3'], grads['b3'] = affine_backward(dout, scores_cache)

      daffine_relu = relu_backward(daffine, affine_relu_cache)
    
      dbnorm2, grads['gamma2'], grads['beta2'] = batchnorm_backward(daffine_relu, batchnorm2_cache)

      daffine, grads['W2'], grads['b2'] = affine_backward(dbnorm2, affine_cache)

      dmax_pool = max_pool_backward_fast(daffine, max_cache)

      drelu = relu_backward(dmax_pool, conv_relu_cache)
    
      dbnorm1, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(drelu, batchnorm1_cache)

      dX, grads['W1'], grads['b1'] = conv_backward_fast(dbnorm1, conv_cache)
    
    else:
      loss, dout = softmax_loss(scores, y)

      loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))

      daffine, grads['W3'], grads['b3'] = affine_backward(dout, scores_cache)

      daffine_relu = relu_backward(daffine, affine_relu_cache)
    
      daffine, grads['W2'], grads['b2'] = affine_backward(daffine_relu, affine_cache)

      dmax_pool = max_pool_backward_fast(daffine, max_cache)

      drelu = relu_backward(dmax_pool, conv_relu_cache)
    
      dX, grads['W1'], grads['b1'] = conv_backward_fast(drelu, conv_cache)

    grads['W3'] += self.reg*W3
    grads['W2'] += self.reg*W2
    grads['W1'] += self.reg*W1
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads