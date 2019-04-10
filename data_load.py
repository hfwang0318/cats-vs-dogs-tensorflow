import tensorflow as tf 
import os 
import numpy as np 
import cv2 
import tensorflow.keras.backend as K

def load_train_data(train_path, sess, batch_size=32, num_epochs=200):
    """ 加载训练数据集
    args:
        train_path: 训练图片存放路径
        sess: 一个 tf.Session() 的实例
        batch_size: 一个 batch 的大小
        num_epochs: 总迭代代数
    return:
        train_iter: 训练集迭代器
        num_examples: 训练集数量
    """

    train_img_paths = []
    train_labels = []

    # 加载训练图片路径
    for img_path in os.listdir(train_path + 'cats/'):
        train_img_paths.append(train_path + 'cats/' + img_path)

    for img_path in os.listdir(train_path + 'dogs/'):
        train_img_paths.append(train_path + 'dogs/' + img_path)
    num_examples = len(train_img_paths)

    for i in range(num_examples):
        if i < num_examples/2:
            train_labels.append([1, 0])
        else:
            train_labels.append([0, 1])


    # 构建数据集
    train_img_paths = np.array(train_img_paths)
    train_labels = np.array(train_labels).astype('float32')

    train_img_paths_ph = tf.placeholder(train_img_paths.dtype, train_img_paths.shape)
    train_labels_ph = tf.placeholder(train_labels.dtype, train_labels.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths_ph, train_labels_ph))
    # train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))

    train_ds = train_ds.repeat(num_epochs) \
                       .shuffle(len(train_img_paths)) \
                       .map(lambda train_img_path, train_label: tf.py_func(
                           load_img,
                           [train_img_path, train_label],
                           [tf.float32, tf.float32]
                        ), num_parallel_calls=6) \
                       .map(lambda img, label: process(img, label)) \
                       .batch(batch_size) \
                       .prefetch(batch_size)

    train_iter = train_ds.make_initializable_iterator()
    # train_iter = train_ds.make_one_shot_iterator()

    # 对数据集进行初始化
    sess.run(train_iter.initializer, 
        feed_dict={train_img_paths_ph: train_img_paths, train_labels_ph: train_labels})

    return [train_iter, num_examples]


def load_val_data(val_path, sess, batch_size):
    """ 加载验证数据集
    args:
        val_path: 验证数据集目录
        sess: 一个 tf.Session() 实例
        batch_size: 一个 batch 的大小
    returns:
        val_iter: 验证集的迭代器
        num_exampls: 验证集数量
    """

    val_img_paths = []
    val_labels = []

    # 加载验证图片路径
    for img_path in os.listdir(val_path + 'cats/'):
        val_img_paths.append(val_path + 'cats/' + img_path)

    for img_path in os.listdir(val_path + 'dogs/'):
        val_img_paths.append(val_path + 'dogs/' + img_path)
    num_examples = len(val_img_paths)

    for i in range(len(val_img_paths)):
        if i < len(val_img_paths)/2:
            val_labels.append([1, 0])
        else:
            val_labels.append([0, 1])

    val_img_paths = np.array(val_img_paths)
    val_labels = np.array(val_labels).astype('float32')

    val_img_paths_ph = tf.placeholder(val_img_paths.dtype, val_img_paths.shape)
    val_labels_ph = tf.placeholder(val_labels.dtype, val_labels.shape)

    val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths_ph, val_labels_ph))

    val_ds = val_ds.repeat(1) \
                   .map(lambda val_img_path, val_label: tf.py_func(
                       load_img,
                       [val_img_path, val_label],
                       [tf.float32, tf.float32]
                    ), num_parallel_calls=6) \
                   .map(lambda img, label: process(img, label)) \
                   .batch(batch_size) \
                   .prefetch(batch_size)

    val_iter = val_ds.make_initializable_iterator()

    sess.run(val_iter.initializer, 
        feed_dict={val_img_paths_ph: val_img_paths, val_labels_ph: val_labels})

    return [val_iter, num_examples]


def load_img(img_path, *args):
    img_path = img_path.decode()
    img = cv2.imread(img_path)[:,:,(2,1,0)].astype('float32')
    return img, args[0]


def process(img, *args):
    # img = tf.nn.l2_normalize(img)
    img = tf.image.per_image_standardization(img) # 标准化
    img = tf.image.resize_image_with_crop_or_pad(img, 224, 224)
    return img, args[0]
