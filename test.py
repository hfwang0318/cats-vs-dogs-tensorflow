import tensorflow as tf 
import numpy as np
import tensorflow.keras.backend as K
import net
import data_load
import cv2


def main():
    test_path = 'val/dogs/dog.10010.jpg'
    checkpoint_dir = 'checkpoint/'

    inp = tf.placeholder(tf.float32, (None, 224, 224, 3))
    out, _, _ = net.full_net(inp)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            print('loaded', ckpt)
            saver.restore(sess, ckpt)

        img = cv2.imread(test_path)
        img = cv2.resize(img, (224, 224)) / 255.
        img = np.reshape(img, (1, 224, 224, 3))

        print('start test......')
        result = sess.run(out, feed_dict={inp: img})
        if np.argmax(result[0]) == 0:
            print('cat')
        else:
            print('dog')
        index = np.argmax(result[0])
        value = result[0][index]
        prob = sess.run(tf.nn.softmax([value]))[0]
        print('prob {}'.format(prob))

if __name__ == '__main__':
    main()