from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNetArqui12(object):
    """
    A convolutional network with the following architecture:

    [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, hidden_dims_conv, hidden_dims_aff, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        
        new:
            hidden_dims_conv, 
            hidden_dims_aff

        """

        self.params = {}
        self.params2 = {}
        self.reg = reg
        self.dtype = dtype

        self.num_layers_conv = len(hidden_dims_conv)
        self.num_layers_aff = len(hidden_dims_aff)

        C, H, W = input_dim

        # [conv-relu-pool]xN + [conv-relu]
        num_channels, new_H, new_W = C, H, W
        for i,j in enumerate(range(len(hidden_dims_conv)), start=1):

            num_filters, filter_size, conv_stride, pool_dim, pool_stride = hidden_dims_conv[j]

            self.params['W%d'%i] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_channels, filter_size, filter_size))
            self.params['b%d'%i] = np.zeros(num_filters)
            self.params2['conv_param%d'%i] = {'stride': conv_stride, 'pad': (filter_size - 1) // 2}
            self.params2['pool_param%d'%i] = {'pool_height': pool_dim[0], 'pool_width': pool_dim[1], 'stride': pool_stride}

            if self.params2['pool_param%d'%i]['stride'] != 0:
                new_H, new_W = (new_H // pool_dim[0]), (new_W // pool_dim[1])
            num_channels = num_filters

        # [affine]xM
        l = len(hidden_dims_conv) + 1
        previous_dim = num_filters*new_H*new_W
        for i,j in enumerate(range(len(hidden_dims_aff)), start=l):
            self.params['W%d'%i] = np.random.normal(loc=0.0, scale=weight_scale, size=(previous_dim, hidden_dims_aff[j]))
            self.params['b%d'%i] = np.zeros(hidden_dims_aff[j])

            previous_dim = hidden_dims_aff[j]

        # output layer
        self.params['W%d'%(self.num_layers_conv + self.num_layers_aff + 1)] = np.random.normal(loc=0.0, scale=weight_scale, size=(previous_dim, num_classes))
        self.params['b%d'%(self.num_layers_conv + self.num_layers_aff + 1)] = np.zeros(num_classes)
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        ### FOWARD PASS ###
        scores, cache, reg_sum = None, {}, 0

        for i in range(1, self.num_layers_conv):
            X, cache[i] = conv_relu_pool_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i],  self.params2['pool_param%d'%i])
            reg_sum += np.sum(self.params['W%d'%i] ** 2)

        i = self.num_layers_conv
        if self.params2['pool_param%d'%i]['stride'] != 0:
            X, cache[i] = conv_relu_pool_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i],  self.params2['pool_param%d'%i])
        else:
            X, cache[i] = conv_relu_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i])

        reg_sum += np.sum(self.params['W%d'%i] ** 2)

        for i in range(self.num_layers_conv+1, self.num_layers_conv+1+self.num_layers_aff):
            X, cache[i]  = affine_relu_forward(X, self.params['W%d'%i], self.params['b%d'%i])
            reg_sum += np.sum(self.params['W%d'%i] ** 2)

        out_layer = self.num_layers_conv + self.num_layers_aff + 1
        scores, cache[out_layer] = affine_forward(X, self.params['W%d'%out_layer], self.params['b%d'%out_layer])


        if y is None:
            return scores

        ### BACKWARD PASS ###
        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)

        # Add regularization
        loss += self.reg * 0.5 * reg_sum

        dX, grads['W%d'%out_layer], grads['b%d'%out_layer] = affine_backward(dscores, cache[out_layer])
        
        for i in range(self.num_layers_aff+self.num_layers_conv, self.num_layers_conv, -1):
            dX, grads['W%d'%i], grads['b%d'%i] = affine_relu_backward(dX, cache[i])
        
        i = self.num_layers_conv
        if self.params2['pool_param%d'%i]['stride'] != 0:
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_pool_backward(dX, cache[i])
        else:
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_backward(dX, cache[i])

        for i in range(self.num_layers_conv-1, 0, -1):        
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_pool_backward(dX, cache[i])

        for i in range(self.num_layers_conv + self.num_layers_aff + 1, 0, -1):
            grads['W%d'%i] += self.reg * self.params['W%d'%i]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads




class ConvNetArqui3(object):
    """
    A convolutional network with the following architecture:

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, hidden_dims_conv, hidden_dims_aff, use_batchnorm, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        
        new:
            hidden_dims_conv, 
            hidden_dims_aff

        """

        self.params = {}
        self.params2 = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim

        # [conv-relu-pool]xN
        num_channels, new_H, new_W = C, H, W
        for i,j in enumerate(range(len(hidden_dims_conv)), start=1):

            num_filters, filter_size, conv_stride, pool_dim, pool_stride = hidden_dims_conv[j]

            self.params['W%d'%i] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_channels, filter_size, filter_size))
            self.params['b%d'%i] = np.zeros(num_filters)
            self.params2['conv_param%d'%i] = {'stride': conv_stride, 'pad': (filter_size - 1) // 2}
            self.params2['pool_param%d'%i] = {'pool_height': pool_dim[0], 'pool_width': pool_dim[1], 'stride': pool_stride}

            if self.params2['pool_param%d'%i]['stride'] != 0:
                new_H, new_W = (new_H // pool_dim[0]), (new_W // pool_dim[1])
            num_channels = num_filters
        
        # conv - relu 
        '''
        i = self.num_layers_conv
        num_filters, filter_size, conv_stride, pool_dim, pool_stride  = hidden_dims_conv[i-1]

        self.params['W%d'%i] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_channels, filter_size, filter_size))
        self.params['b%d'%i] = np.zeros(num_filters)
        self.params2['conv_param%d'%i] = {'stride': conv_stride, 'pad': (filter_size - 1) // 2}
        '''

        # [affine]xM
        l = len(hidden_dims_conv) + 1
        previous_dim = num_filters*new_H*new_W
        for i,j in enumerate(range(len(hidden_dims_aff)), start=l):
            self.params['W%d'%i] = np.random.normal(loc=0.0, scale=weight_scale, size=(previous_dim, hidden_dims_aff[j]))
            self.params['b%d'%i] = np.zeros(hidden_dims_aff[j])

            previous_dim = hidden_dims_aff[j]

        # output layer
        self.params['W%d'%(self.num_layers_conv + self.num_layers_aff + 1)] = np.random.normal(loc=0.0, scale=weight_scale, size=(previous_dim, num_classes))
        self.params['b%d'%(self.num_layers_conv + self.num_layers_aff + 1)] = np.zeros(num_classes)
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        ### FOWARD PASS ###
        scores, cache, reg_sum = None, {}, 0

        for i in range(1, self.num_layers_conv):
            X, cache[i] = conv_relu_pool_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i],  self.params2['pool_param%d'%i])
            reg_sum += np.sum(self.params['W%d'%i] ** 2)

        i = self.num_layers_conv
        if self.params2['pool_param%d'%i]['stride'] != 0:
            X, cache[i] = conv_relu_pool_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i],  self.params2['pool_param%d'%i])
        else:
            X, cache[i] = conv_relu_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i])

        reg_sum += np.sum(self.params['W%d'%i] ** 2)

        for i in range(self.num_layers_conv+1, self.num_layers_conv+1+self.num_layers_aff):
            X, cache[i]  = affine_relu_forward(X, self.params['W%d'%i], self.params['b%d'%i])
            reg_sum += np.sum(self.params['W%d'%i] ** 2)

        out_layer = self.num_layers_conv + self.num_layers_aff + 1
        scores, cache[out_layer] = affine_forward(X, self.params['W%d'%out_layer], self.params['b%d'%out_layer])


        if y is None:
            return scores

        ### BACKWARD PASS ###
        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)

        # Add regularization
        loss += self.reg * 0.5 * reg_sum

        dX, grads['W%d'%out_layer], grads['b%d'%out_layer] = affine_backward(dscores, cache[out_layer])
        
        for i in range(self.num_layers_aff+self.num_layers_conv, self.num_layers_conv, -1):
            dX, grads['W%d'%i], grads['b%d'%i] = affine_relu_backward(dX, cache[i])
        
        i = self.num_layers_conv
        if self.params2['pool_param%d'%i]['stride'] != 0:
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_pool_backward(dX, cache[i])
        else:
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_backward(dX, cache[i])

        for i in range(self.num_layers_conv-1, 0, -1):        
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_pool_backward(dX, cache[i])

        for i in range(self.num_layers_conv + self.num_layers_aff + 1, 0, -1):
            grads['W%d'%i] += self.reg * self.params['W%d'%i]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNetArqui12(object):
    """
    A convolutional network with the following architecture:

    [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, hidden_dims_conv, hidden_dims_aff, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        
        new:
            hidden_dims_conv, 
            hidden_dims_aff

        """

        self.params = {}
        self.params2 = {}
        self.reg = reg
        self.dtype = dtype

        self.num_layers_conv = len(hidden_dims_conv)
        self.num_layers_aff = len(hidden_dims_aff)

        C, H, W = input_dim

        # [conv-relu-pool]xN + [conv-relu]
        num_channels, new_H, new_W = C, H, W
        for i,j in enumerate(range(len(hidden_dims_conv)), start=1):

            num_filters, filter_size, conv_stride, pool_dim, pool_stride = hidden_dims_conv[j]

            self.params['W%d'%i] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_channels, filter_size, filter_size))
            self.params['b%d'%i] = np.zeros(num_filters)
            self.params2['conv_param%d'%i] = {'stride': conv_stride, 'pad': (filter_size - 1) // 2}
            self.params2['pool_param%d'%i] = {'pool_height': pool_dim[0], 'pool_width': pool_dim[1], 'stride': pool_stride}

            if self.params2['pool_param%d'%i]['stride'] != 0:
                new_H, new_W = (new_H // pool_dim[0]), (new_W // pool_dim[1])
            num_channels = num_filters

        # [affine]xM
        l = len(hidden_dims_conv) + 1
        previous_dim = num_filters*new_H*new_W
        for i,j in enumerate(range(len(hidden_dims_aff)), start=l):
            self.params['W%d'%i] = np.random.normal(loc=0.0, scale=weight_scale, size=(previous_dim, hidden_dims_aff[j]))
            self.params['b%d'%i] = np.zeros(hidden_dims_aff[j])

            previous_dim = hidden_dims_aff[j]

        # output layer
        self.params['W%d'%(self.num_layers_conv + self.num_layers_aff + 1)] = np.random.normal(loc=0.0, scale=weight_scale, size=(previous_dim, num_classes))
        self.params['b%d'%(self.num_layers_conv + self.num_layers_aff + 1)] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        ### FOWARD PASS ###
        scores, cache, reg_sum = None, {}, 0

        for i in range(1, self.num_layers_conv):
            X, cache[i] = conv_relu_pool_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i],  self.params2['pool_param%d'%i])
            reg_sum += np.sum(self.params['W%d'%i] ** 2)

        i = self.num_layers_conv
        if self.params2['pool_param%d'%i]['stride'] != 0:
            X, cache[i] = conv_relu_pool_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i],  self.params2['pool_param%d'%i])
        else:
            X, cache[i] = conv_relu_forward(X, self.params['W%d'%i],  self.params['b%d'%i],  self.params2['conv_param%d'%i])

        reg_sum += np.sum(self.params['W%d'%i] ** 2)

        for i in range(self.num_layers_conv+1, self.num_layers_conv+1+self.num_layers_aff):
            X, cache[i]  = affine_relu_forward(X, self.params['W%d'%i], self.params['b%d'%i])
            reg_sum += np.sum(self.params['W%d'%i] ** 2)

        out_layer = self.num_layers_conv + self.num_layers_aff + 1
        scores, cache[out_layer] = affine_forward(X, self.params['W%d'%out_layer], self.params['b%d'%out_layer])


        if y is None:
            return scores

        ### BACKWARD PASS ###
        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)

        # Add regularization
        loss += self.reg * 0.5 * reg_sum

        dX, grads['W%d'%out_layer], grads['b%d'%out_layer] = affine_backward(dscores, cache[out_layer])
        
        for i in range(self.num_layers_aff+self.num_layers_conv, self.num_layers_conv, -1):
            dX, grads['W%d'%i], grads['b%d'%i] = affine_relu_backward(dX, cache[i])
        
        i = self.num_layers_conv
        if self.params2['pool_param%d'%i]['stride'] != 0:
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_pool_backward(dX, cache[i])
        else:
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_backward(dX, cache[i])

        for i in range(self.num_layers_conv-1, 0, -1):        
            dX, grads['W%d'%i], grads['b%d'%i] = conv_relu_pool_backward(dX, cache[i])

        for i in range(self.num_layers_conv + self.num_layers_aff + 1, 0, -1):
            grads['W%d'%i] += self.reg * self.params['W%d'%i]

        return loss, grads




class ConvNetArqui3(object):
    """
    A convolutional network with the following architecture:

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, hidden_dims, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        
        self.params = {}
        self.params2 = {}
        self.bn_param = []
        
        self.reg = reg
        self.dtype = dtype

        filter_size = 3

        C, H, W = input_dim
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[0], C, filter_size, filter_size))
        self.params['b1'] = np.zeros(hidden_dims[0])
        self.params['gamma1'],  self.params['beta1'] =  np.ones([hidden_dims[0]]), np.zeros([hidden_dims[0]])
        self.params2['conv_param1'] = {'stride': 1, 'pad': (filter_size - 1) // 2}

        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[1], hidden_dims[0], filter_size, filter_size)) 
        self.params['b2'] = np.zeros(hidden_dims[1])
        self.params['gamma2'],  self.params['beta2'] =  np.ones([hidden_dims[1]]), np.zeros([hidden_dims[1]])
        self.params2['conv_param2'] = {'stride': 1, 'pad': (filter_size - 1) // 2}

        self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[2], hidden_dims[1], filter_size, filter_size))
        self.params['b3'] = np.zeros(hidden_dims[2])
        self.params['gamma3'],  self.params['beta3'] =  np.ones([hidden_dims[2]]), np.zeros([hidden_dims[2]])
        self.params2['conv_param3'] = {'stride': 1, 'pad': (filter_size - 1) // 2}  

        self.params2['pool_param4'] = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        self.params['W5'] = np.random.normal(loc=0.0, scale=weight_scale, size=((H // 2)*(W // 2)*hidden_dims[2], hidden_dims[3])) 
        self.params['b5'] = np.zeros(hidden_dims[3])

        self.params['gamma6'],  self.params['beta6'] =  np.ones([hidden_dims[3]]), np.zeros([hidden_dims[3]])

        # output layer.
        self.params['W7'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[3], num_classes)) 
        self.params['b7'] = np.zeros(num_classes)

        num_layers = 7
        self.bn_param = [{'mode': 'train'} for i in range(num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):

        mode = 'test' if y is None else 'train'
        for bn in self.bn_param:
            bn['mode'] = mode


        # Forward pass.
        cache = {}
        X, cache[1] = conv_bn_relu_forward(X, self.params['W1'],  self.params['b1'],  self.params['gamma1'],  self.params['beta1'], self.params2['conv_param1'], self.bn_param[0])
        X, cache[2] = conv_bn_relu_forward(X, self.params['W2'],  self.params['b2'],  self.params['gamma2'],  self.params['beta2'], self.params2['conv_param2'], self.bn_param[1])
        X, cache[3] = conv_bn_relu_forward(X, self.params['W3'],  self.params['b3'],  self.params['gamma3'],  self.params['beta3'], self.params2['conv_param3'], self.bn_param[2])
        X, cache[4] = max_pool_forward_fast(X, self.params2['pool_param4'])
        X, cache[5] = affine_relu_forward(X, self.params['W5'], self.params['b5'])
        X, cache[6] = batchnorm_forward(X, self.params['gamma6'], self.params['beta6'], self.bn_param[5])
        scores, cache[7] = affine_forward(X, self.params['W7'], self.params['b7'])
        
        # Loss
        loss, dscores = softmax_loss(scores, y)

        if y is None:
            return scores
        
        # Regularization loss
        reg_sum = 0
        reg_sum += np.sum(self.params['W1'] ** 2)
        reg_sum += np.sum(self.params['W2'] ** 2)
        reg_sum += np.sum(self.params['W3'] ** 2)
        reg_sum += np.sum(self.params['W5'] ** 2)
        reg_sum += np.sum(self.params['W7'] ** 2)
        loss += self.reg * 0.5 * reg_sum

        # Backward
        grads = {}
        dX, grads['W7'], grads['b7']        = affine_backward(dscores, cache[7])
        dX, grads['gamma6'], grads['beta6'] = batchnorm_backward(dX, cache[6])
        dX, grads['W5'], grads['b5']        = affine_relu_backward(dX, cache[5])
        dX                                  = max_pool_backward_fast(dX, cache[4])
        dX, grads['W3'], grads['b3'], grads['gamma3'], grads['beta3'] = conv_bn_relu_backward(dX, cache[3])
        dX, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_bn_relu_backward(dX, cache[2])
        dX, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_backward(dX, cache[1])

        # Regularization gradient
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']
        grads['W5'] += self.reg * self.params['W5']
        grads['W7'] += self.reg * self.params['W7']


        return loss, grads




class ConvNetArqui4(object):
    """
    A convolutional network with the following architecture:

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, hidden_dims, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        
        self.params = {}
        self.params2 = {}
        self.bn_param = []
        
        self.reg = reg
        self.dtype = dtype

        filter_size = 3

        C, H, W = input_dim
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[0], C, filter_size, filter_size))
        self.params['b1'] = np.zeros(hidden_dims[0])
        self.params['gamma1'],  self.params['beta1'] =  np.ones([hidden_dims[0]]), np.zeros([hidden_dims[0]])
        self.params2['conv_param1'] = {'stride': 1, 'pad': (filter_size - 1) // 2}

        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[1], hidden_dims[0], filter_size, filter_size)) 
        self.params['b2'] = np.zeros(hidden_dims[1])
        self.params['gamma2'],  self.params['beta2'] =  np.ones([hidden_dims[1]]), np.zeros([hidden_dims[1]])
        self.params2['conv_param2'] = {'stride': 1, 'pad': (filter_size - 1) // 2}

        self.params2['pool_param4'] = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        self.params['W5'] = np.random.normal(loc=0.0, scale=weight_scale, size=((H // 2)*(W // 2)*hidden_dims[1], hidden_dims[2])) 
        self.params['b5'] = np.zeros(hidden_dims[2])

        self.params['gamma6'],  self.params['beta6'] =  np.ones([hidden_dims[2]]), np.zeros([hidden_dims[2]])

        # output layer.
        self.params['W7'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[2],num_classes)) 
        self.params['b7'] = np.zeros(num_classes)

        num_layers = 7
        self.bn_param = [{'mode': 'train'} for i in range(num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):

        mode = 'test' if y is None else 'train'
        for bn in self.bn_param:
            bn['mode'] = mode


        # Forward pass.
        cache = {}
        X, cache[1] = conv_bn_relu_forward(X, self.params['W1'],  self.params['b1'],  self.params['gamma1'],  self.params['beta1'], self.params2['conv_param1'], self.bn_param[0])
        X, cache[2] = conv_bn_relu_forward(X, self.params['W2'],  self.params['b2'],  self.params['gamma2'],  self.params['beta2'], self.params2['conv_param2'], self.bn_param[1])
        X, cache[4] = max_pool_forward_fast(X, self.params2['pool_param4'])
        X, cache[5] = affine_relu_forward(X, self.params['W5'], self.params['b5'])
        X, cache[6] = batchnorm_forward(X, self.params['gamma6'], self.params['beta6'], self.bn_param[5])
        scores, cache[7] = affine_forward(X, self.params['W7'], self.params['b7'])
        
        # Loss
        loss, dscores = softmax_loss(scores, y)

        if y is None:
            return scores
        
        # Regularization loss
        reg_sum = 0
        reg_sum += np.sum(self.params['W1'] ** 2)
        reg_sum += np.sum(self.params['W2'] ** 2)
        reg_sum += np.sum(self.params['W5'] ** 2)
        reg_sum += np.sum(self.params['W7'] ** 2)
        loss += self.reg * 0.5 * reg_sum

        # Backward
        grads = {}
        dX, grads['W7'], grads['b7']        = affine_backward(dscores, cache[7])
        dX, grads['gamma6'], grads['beta6'] = batchnorm_backward(dX, cache[6])
        dX, grads['W5'], grads['b5']        = affine_relu_backward(dX, cache[5])
        dX                                  = max_pool_backward_fast(dX, cache[4])
        dX, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_bn_relu_backward(dX, cache[2])
        dX, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_backward(dX, cache[1])

        # Regularization gradient
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W5'] += self.reg * self.params['W5']
        grads['W7'] += self.reg * self.params['W7']


        return loss, grads
