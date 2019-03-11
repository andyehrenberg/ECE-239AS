import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape
  F, _, H_f, W_f = w.shape
  H_new = int(1+(H+2*pad-H_f)/stride)
  W_new = int(1+(W+2*pad-W_f)/stride)

  out = np.zeros((N, F, H_new, W_new))

  x = np.pad(x, ((0,0),(0,0),(pad, pad),(pad, pad)), mode='constant')

  for i in range(N):
    for z in range(F):
      for j in range(H_new):
        for k in range(W_new):
          vals = x[i,:,j*stride:(j*stride+H_f),k*stride:(k*stride+W_f)]*w[z,:,:,:]
          out[i, z, j, k] = np.sum(vals)+b[z]        
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N, C, H, W = x.shape
  F, _, H_f, W_f = w.shape

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  for i in range(N):
    for z in range(num_filts):
      for j in range(out_height):
        h_start = j*stride
        for k in range(out_width):
          w_start = k*stride
          dx[i,:,h_start:(h_start+f_height),w_start:(w_start+f_width)] += w[z,:,:,:]*dout[i,z,j,k]
          dw[z,:,:,:] += x[i,:,h_start:(h_start+f_height),w_start:(w_start+f_width)]*dout[i,z,j,k]
            
  dx = dx[:,:,pad:-pad,pad:-pad]

  db = dout.sum(axis=(0,2,3))
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  h_pool = pool_param['pool_height']
  w_pool = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  h_out = int((H - h_pool)/stride + 1)
  w_out = int((W - w_pool)/stride + 1)
  
  out = np.zeros((N, C, h_out, w_out))

  for h in np.arange(h_out):
    for w in np.arange(w_out):
      h_start = h*stride
      w_start = w*stride
      h_end = h*stride + h_pool
      w_end = w*stride + w_pool
      out[:,:,h,w] = x[:,:,h_start:h_end,w_start:w_end].max(axis=(2,3))     
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  H_out = int((H-pool_height)/stride+1)
  W_out = int((W-pool_width)/stride+1)
  dx = np.zeros_like(x)
  for n in range(N):
    for c in range(C):
      for h in range(H_out):
        for w in range(W_out):
          dslice = np.zeros((pool_height,pool_width))
          input_slice = x[n,c,h*stride:(h*stride+pool_height),w*stride:(w*stride+pool_width)]
          maxes = np.where(input_slice==input_slice.max())
          dslice[maxes[0], maxes[1]] = dout[n,c,h,w]
          dx[n,c,h*stride:(h*stride+pool_height),w*stride:(w*stride+pool_width)] += dslice
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   A few steps here:
    #     (1) Calculate the running mean and variance of the minibatch.
    #     (2) Normalize the activations with the batch mean and variance.
    #     (3) Scale and shift the normalized activations.  Store this
    #         as the variable 'out'
    #     (4) Store any variables you may need for the backward pass in
    #         the 'cache' variable.
    # ================================================================ #
    xbar = np.mean(x, axis=0)
    var = np.var(x, axis=0)    
    
    xstand = (x - xbar)/np.sqrt(var + eps) 
    out = gamma*xstand+beta
    
    running_mean = momentum * running_mean + (1 - momentum) * xbar
    running_var = momentum * running_var + (1 - momentum) * var

    cache = (xstand, xbar, var, eps, gamma, beta, x)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
  
  elif mode == 'test':
        
    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the testing time normalized activations.  Normalize using
    #   the running mean and variance, and then scale and shift appropriately.
    #   Store the output as 'out'.
    # ================================================================ #
    xstand = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * xstand + beta
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  x = np.swapaxes(x,0,1)
  out, cache = batchnorm_forward(x.reshape(C,N*H*W).T, gamma, beta, bn_param)
  out = out.T.reshape(C,N,H,W).swapaxes(0,1)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache

def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the batchnorm backward pass, calculating dx, dgamma, and dbeta.
  # ================================================================ #
  xstand, xbar, var, eps, gamma, beta, x = cache
  N, D = dout.shape

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * xstand, axis=0)

  dxstand = dout * gamma
  dxbar1 = dxstand * 1 / np.sqrt(var + eps)
  divar = np.sum(dxstand * (x - xbar), axis=0)
  dvar = divar * -1 / 2 * (var + eps) ** (-3/2)
  dsq = 1 / N * np.ones((N, D)) * dvar
  dxbar2 = 2 * (x - xbar) * dsq
  dx1 = dxbar1 + dxbar2
  dxbar = -1 * np.sum(dxbar1 + dxbar2, axis=0)
  dx2 = 1 / N * np.ones((N, D)) * dxbar
  dx = dx1 + dx2
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
  
  return dx, dgamma, dbeta

def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dx, dgamma, dbeta = batchnorm_backward(dout.swapaxes(0,1).reshape(C,-1).T, cache)
  dx = dx.T.reshape(C,N,H,W).swapaxes(0,1)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta