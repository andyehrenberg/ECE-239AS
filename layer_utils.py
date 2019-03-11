from .layers import *
from cs231n.fast_layers import *
from .conv_layers import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def affine_relu_forward(x, w, b):
  """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """Convenience layer that performs a convolution, spatial
  batchnorm, a ReLU, and a pool.
  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer
  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
  norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
  relu, relu_cache = relu_forward(norm)
  out, pool_cache = max_pool_forward_fast(relu, pool_param)

  cache = (conv_cache, norm_cache, relu_cache, pool_cache)

  return out, cache

def conv_norm_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, norm_cache, relu_cache, pool_cache = cache

  dpool = max_pool_backward_fast(dout, pool_cache)
  drelu = relu_backward(dpool, relu_cache)
  dnorm, dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
  dx, dw, db = conv_backward_fast(dnorm, conv_cache)

  return dx, dw, db, dgamma, dbeta

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta : Weight for the batch norm regularization
  - bn_params : Contain variable use to batch norml, running_mean and var
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """

  h, h_cache = affine_forward(x, w, b)
  hnorm, hnorm_cache = batchnorm_forward(h, gamma, beta, bn_param)
  hnormrelu, relu_cache = relu_forward(hnorm)
  cache = (h_cache, hnorm_cache, relu_cache)

  return hnormrelu, cache

def affine_norm_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  h_cache, hnorm_cache, relu_cache = cache

  dhnormrelu = relu_backward(dout, relu_cache)
  dhnorm, dgamma, dbeta = batchnorm_backward(dhnormrelu, hnorm_cache)
  dx, dw, db = affine_backward(dhnorm, h_cache)

  return dx, dw, db, dgamma, dbeta