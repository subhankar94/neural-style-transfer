import tensorflow as tf
import numpy as np
import scipy
import cv2

vgg = scipy.io.loadmat(vgg_data_path)

def vgg_conv_layer(input, vgg_weights, idx):
    weights, bias = vgg_weights[idx][0][0][0][0]
    weights = np.transpose(weights, (1, 0, 2, 3))
    bias    = bias.reshape(-1)
    conv    = tf.nn.conv2d(input,
                tf.constant(weights),
                strides=(1, 1, 1, 1),
                padding='SAME')
    return tf.nn.bias_add(conv, bias)

def vgg_pool_layer(input):
    return tf.nn.max_pool(input,
             ksize  =(1, 2, 2, 1),
             strides=(1, 2, 2, 1),
             padding='SAME')

def relu_layer(input):
    return tf.nn.relu(input)


def vgg_net(input_image):

    # extract net data
    vgg_weights    = vgg['layers'][0]
    vgg_mean       = vgg['normalization'][0][0][0]
    vgg_mean_pixel = np.mean(mean, axis=(0, 1))

    # list layers to build net
    vgg_layers = [
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2',
        'pool1',

        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'pool2',

        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3',
        'conv3_4', 'relu3_4',
        'pool3',

        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3',
        'conv4_4', 'relu4_4',
        'pool4',

        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3',
        'conv5_4', 'relu5_4'
    ]

    # loop through layers to build network
    vgg_net    = {}
    curr_layer = input_image

    for idx, layer_name in enumerate(vgg_layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            curr_layer = vgg_conv_layer(curr_layer, vgg_weights, idx)
        elif layer_type == 'relu':
            curr_layer = relu_layer(curr_layer)
        elif layer_type == 'pool':
            curr_layer = vgg_pool_layer(curr_layer)
        vgg_net[layer_name] = curr_layer

    return vgg_net, vgg_mean_pixel

def inorm_net(image):
    conv1 = inorm_conv_layer(image, 32, 9, 1)
    relu1 = relu_layer(conv1)
    conv2 = inorm_conv_layer(relu1, 64, 3, 2)
    relu2 = relu_layer(conv2)
    conv3 = inorm_conv_layer(relu2, 128, 3, 2)
    relu3 = relu_layer(conv3)
    resid1 = inorm_residual_block(relu3, 3)
    resid2 = inrom_residual_block(resid1, 3)
    resid3 = inrom_residual_block(resid2, 3)
    resid4 = inrom_residual_block(resid3, 3)
    resid5 = inrom_residual_block(resid4, 3)
    conv_t1 = inrom_conv_tranpose_layer(resid5, 64, 3, 2)
    relu_t1 = relu_layer(conv_t1)
    conv_t2 = inrom_conv_tranpose_layer(relu_t1, 32, 3, 2)
    relu_t2 = relu_layer(conv_t2)
    conv_t3 = _conv_layer(relu_t2, 3, 9, 1)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def inrom_conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = inorm_conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = inrom_instance_norm(net)

    return net

def inrom_conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = inrom_conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.pack(new_shape)
    strides_shape = [1,strides,strides,1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = inrom_instance_norm(net)
    return tf.nn.relu(net)

def inrom_residual_block(net, filter_size=3):
    tmp = inrom_conv_layer(net, 128, filter_size, 1)
    relu_tmp = relu_layer(tmp)
    return net + inrom_conv_layer(relu_tmp, 128, filter_size, 1)

def inrom_instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def inrom_conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32)
    return weights_init

def content_loss(p, x):
    loss = 1/2 * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

def style_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G
