import numpy as np, sys, os, scipy.misc, tensorflow as tf, style-transfer

def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    img_shape = get_img(data_in[0]).shape
    batch_size = min(len(paths_out), batch_size)
    cur_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Graph().device(device_t),
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, ) + img_shape
        img_placeholder = tf.placeholder(
                                tf.float32,
                                shape=batch_shape,
                                name='img_placeholder'
                                )
        preds = inorm_net(img_placeholder)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            cur_batch_out = paths_out[pos:pos+batch_size]
            cur_batch_in  = data_in[pos:pos+batch_size]
            X = np.zeros(batch_shape, dtype=np.float32)
            for idx, path_in in enumerate(curr_batch_in):
                img = get_img(path_in)
                X[idx] = img

            _preds = sess.run(preds, feed_dict{img_placeholder:X})
            for idx, path_out in enumerate(cur_batch_out):
                save_img(path_out, _preds[idx])

        remaining_in  = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]

    if len(remaining_in) > 0:
        ffwd(remaining_in,
            remaining_out,
            checkpoint_dir,
            device_t=device_t,
            batch_size=1)

def main():
    
