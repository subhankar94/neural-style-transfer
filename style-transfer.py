import numpy as np, tensorflow as tf, scipy, cv2, sys, os
from functools import reduce
from argparse import ArgumentParser

vgg_data_path = ''
vgg = scipy.io.loadmat(vgg_data_path)
vgg_mean_pixel = np.mean(vgg['normalization'][0][0][0], axis=(0, 1))
style_layers = ['relu1_1',
                'relu2_1',
                'relu3_1',
                'relu4_1',
                'relu5_1']
content_layer = 'relu4_2'

def save_img(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

def scale_img(path, scale):
    scale = float(scale)
    h, w, d = scipy.misc.imread(path, mode='RGB').shape
    shape = (int(h*scale), int(w*scale), d)
    return get_img(path, img_size=new_shape)

def get_img(path, img_size=False):
    img = scipy.misc.imread(path, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

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

    return vgg_net

def inrom_conv_layer(net, num_filters, filter_size, strides):
    weights_init = inorm_conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = inrom_instance_norm(net)
    return net

def inrom_conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = inrom_conv_init_vars(net,
                    num_filters,
                    filter_size,
                    transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.pack(new_shape)
    strides_shape = [1,strides,strides,1]

    net = tf.nn.conv2d_transpose(net,
            weights_init,
            tf_shape,
            strides_shape,
            padding='SAME')
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

    weights_init = tf.Variable(tf.truncated_normal
                    (weights_shape, stddev=0.1, seed=1),
                    dtype=tf.float32)
    return weights_init

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

def get_content_loss(weight, net, features, batch_size):
    content_size = reduce(
                        np.multiply,
                        (dim.value for dim in features.get_shape()[1:]),
                        1
                    )*batch_size
    loss = weight * (2 * tf.nn.l2_loss(net - features)/content_size)
    return loss

def get_style_loss(weight, generated, originals, batch_size):
    style_losses = []
    for idx in range(len(originals)):
        b, h, w, d = generated[idx].get_shape()
        G = gram_matrix(generated[idx], b.value, h.value, w.value, d.value)
        A = originals[idx]
        layer_loss = 2 * tf.nn.l2_loss(G - A)/A.size
    loss = weight * reduce(tf.add, style_losses)/batch_size
    return loss

def gram_matrix(x, b, h, w, d):
    F = tf.reshape(x, (b, h * w, d))
    G = tf.batch_matmul(tf.transpose(F, perm=[0,2,1]), F)/(h * w * f)
    return G

def gram_matrix_np(x, area, depth):
    F = np.reshape(x, (area, depth))
    G = np.matmul(tf.transpose(F), F)/F.size
    return G

def optimize(content_targets, style_target,
             content_weight,  style_weight,
             print_iter=1000,
             batch_size=4,
             save_path='saver/fns.ckpt',
             learning_rate=1e-3):

    style_features   = {}
    content_features = {}

    trim = len(contnet_targets)%batch_size
    if trim > 0:
        contnet_targets = contnet_targets[:-trim]

    batch_shape = (batch_size, 256, 256, 3)
    style_shape = (1, ) + style_target.shape

    with tf.Graph().as_default(), tf.Session() as sess:
        style_image = tf.placeholder(
                        tf.float32,
                        shape=style_shape,
                        name='style_image')
        style_image_pre = style_image - vgg_mean_pixel

        net = vgg_net(style_image_pre)
        style_pre = np.array([style_target])
        for layer in style_layers:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            style_features[layer] = gram_matrix_np(
                                        features,
                                        (-1, features.shape[3]))

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(
                            tf.float32,
                            shape=batch_shape,
                            name='X_content')
        X_content_pre = X_content - vgg_mean_pixel

        content_net = vgg_net(X_content_pre)
        content_features[content_layer] = content_net[content_layer]

        preds = inorm_net(X_content/255.0)
        preds_pre = preds - vgg_mean_pixel
        net = vgg_net(preds_pre)

        content_loss = get_content_loss(
                            content_weight,
                            net[content_layer],
                            content_features[content_layer],
                            batch_size)

        style_loss   = get_style_loss(
                            style_weight,
                            [net[l] for l in style_layers],
                            [style_features[l] for l in style_layers],
                            batch_size)

        loss = content_loss + style_loss

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            max_iter, cur_iter = len(content_targets), 0
            while (cur_iter * batch_size) < max_iter:
                b_start = cur_iter * batch_size
                b_end   = b_start + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for idx, img_p in enumerate(content_targets[b_start:b_end]):
                    X_batch[idx] = get_img(
                                        img_p,
                                        (256, 256, 3)
                                    ).astype(np.float32)
                cur_iter += 1
                train_step.run(feed_dict={X_content:X_batch})
                if (int(cur_iter) % print_iter == 0):
                    metrics = [style_loss, content_loss, loss, preds]
                    metrics = sess.run(metrics, feed_dict={X_content:X_batch})
                    _s_loss, _c_loss, _loss, _preds = metrics
                    losses = (_s_loss, _c_loss, _loss)
                    saver = tf.train.Saver()
                    res = saver.save(sess, save_path)
                    yield(_preds, losses, cur_iter, epoch)
