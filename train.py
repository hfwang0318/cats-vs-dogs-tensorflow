import tensorflow as tf 
import tensorflow.keras
import tensorflow.keras.backend as K 
import numpy as np 
import os 
import data_load
import cv2
import net


def main():
    train_path = 'train/' # 训练集路径
    checkpoint_dir = 'checkpoint/' # 模型保存和恢复的路径
    model_name = 'model.ckpt' # 模型名称

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    inp = tf.placeholder(tf.float32, (None, 224, 224, 3))
    labels = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    out, var_init_op, var_init_dict = net.full_net(inp)

    global_step = tf.Variable(0, trainable=False)

    batch_size = 32

    with tf.Session() as sess:
        # 加载数据集
        train_iter, num_train_examples = data_load.load_train_data(
            train_path, sess, batch_size, 100)

        train_img_batch, train_label_batch = train_iter.get_next()
        
        # 得到需要训练的变量列表
        t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'trainable')
        print('trainable variables', t_vars)

        # l2 正则
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), t_vars)
        
        # 交叉熵损失函数 
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=labels)
        loss = tf.reduce_mean(cross_entropy) + reg

        optimizer = tf.train.AdamOptimizer(1e-5).minimize(loss, var_list=t_vars, global_step=global_step)

        # 保存图
        saver = tf.train.Saver(max_to_keep=2)
        
        # 初始化全局变量
        sess.run(tf.global_variables_initializer())

        # 加载模型，若没有模型则用 resnet50 参数初始化
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            print('loaded ' + ckpt)
            saver.restore(sess, ckpt)
        else:
            sess.run(var_init_op, feed_dict=var_init_dict)

        # 每一代的总步数
        steps_per_epoch = num_train_examples // batch_size + 1

        print('start training......')
        while True:
            img_batch, label_batch = sess.run([train_img_batch, train_label_batch])
            
            _, cur_loss, cur_step = sess.run([optimizer, loss, global_step], 
                feed_dict={inp: img_batch, labels: label_batch})
            
            cur_epoch = cur_step // steps_per_epoch + 1

            # 每 500 个 step 保存一次
            if cur_step % 500 == 0:
                saver.save(sess, checkpoint_dir + model_name, global_step=global_step)
                print('model saved')

            # 每 100 个 step 总结一次
            if cur_step % 100 == 0:
                print('epoch %d, step %d, loss %.4f' % (cur_epoch, cur_step, cur_loss))


if __name__ == "__main__":
    main()
