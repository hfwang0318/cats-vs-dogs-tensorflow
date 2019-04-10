import tensorflow as tf 
import numpy as np
import tensorflow.keras.backend as K
import net
import data_load


def main():
    val_path = 'val/' # 验证集路径
    checkpoint_dir = 'checkpoint/' # 加载模型的路径

    input_img = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
    y_true = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    out_img, _, _ = net.full_net(input_img, is_training=False)

    batch_size = 32

    with tf.Session() as sess:
        val_iter, num_examples = data_load.load_val_data(val_path, sess, batch_size=batch_size)
        print('num_examples {}'.format(num_examples))

        saver = tf.train.Saver()
        
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            print('loaded', ckpt)
            saver.restore(sess, ckpt)
        else:
            print('can not find model')
            return
        
        img_batch, label_batch = val_iter.get_next()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_img, labels=y_true))

        total_loss = []
        cur_num = 0 # 当前验证数量
        right = 0 # 预测正确数量
        print('start validation......')
        while True:
            try:
                img, labels = sess.run([img_batch, label_batch])
                cur_loss, outs = sess.run([loss, out_img], feed_dict={input_img: img, y_true: labels})
                total_loss.append(cur_loss)
                for out, label in zip(outs, labels):
                    if np.argmax(out) == np.argmax(label):
                        right += 1

                cur_num += batch_size
                print('process {}/{}'.format(cur_num, num_examples))
            except:
                print('mean loss {} acc {}'.format(np.mean(total_loss), right/cur_num))
                break


if __name__ == '__main__':
    main()
