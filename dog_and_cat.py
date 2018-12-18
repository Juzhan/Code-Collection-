import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.slim import nets
import cv2
slim = tf.contrib.slim

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 3
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCH = 3
is_training = False

data_path = "./train"

# 猫狗大战
# 使用预训练的RestNet50
# 99% 准确，成了
# 怎么写数据 https://blog.csdn.net/pursuit_zhangyu/article/details/80581215
# 哪里的pre https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
# 如何pre https://www.jianshu.com/p/a5ec1dff490d

# 返回二进制的图像信息
def process_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return img.tostring()

# 转换成模板
def convert_to_example(img, label):
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        })
    )
    return example

def load_data():
    train_file = "./dog_and_cat_records/new_train.tfrecords"
    test_file = "./dog_and_cat_records/new_test.tfrecords"

    writer_train = tf.python_io.TFRecordWriter(train_file)
    writer_test = tf.python_io.TFRecordWriter(test_file)

    filenames = os.listdir(data_path)
    np.random.shuffle(filenames)

    print("Loading start =======================")
    for i, filename in enumerate(filenames):
        name = filename[:3]
        filename = os.path.join(data_path, filename)
        img = process_image(filename)
        label = 0
        # label = np.array([0, 1])
        if name == 'cat':
            label = 1
            # label = np.array([1, 0])
        example = convert_to_example(img, label)
        if i % 5 == 0:
            writer_test.write(example.SerializeToString())
        else:
            writer_train.write(example.SerializeToString())
        # 80%
        if i % 1000 == 0:
            print("Loading %d" % i)
    print("Loading end =======================")
    writer_train.close()        
    writer_test.close()

def read_and_decode(example_proto):
    key_features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([2], tf.int64)
    }
    # 解析
    features = tf.parse_single_example(example_proto, key_features)

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(features['label'], tf.int64)
    # print (image)
    return image, label

# 用迭代器获取数据
with tf.variable_scope("TEST_DATA"):
    test_dataset = tf.data.TFRecordDataset('./dog_and_cat_records/test.tfrecords')
    test_dataset = test_dataset.map(read_and_decode)

    test_dataset = test_dataset.batch(100)
    test_iterator = test_dataset.make_initializable_iterator()    
    test_images, test_labels = test_iterator.get_next()

with tf.variable_scope("TRAIN_DATA"):
    dataset = tf.data.TFRecordDataset('./dog_and_cat_records/train.tfrecords')
    dataset = dataset.map(read_and_decode)
    # dataset = dataset.shuffle(dataset)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCH)
    iterator = dataset.make_initializable_iterator()

images, labels = iterator.get_next()

tf_x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
tf_y = tf.placeholder(tf.float32, [None, 2])

# ResNet最后只两个输出
with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(tf_x, num_classes=2,
                                                    is_training=True)
        
with tf.variable_scope('Logits'):
    net = tf.squeeze(net, axis=[1, 2])
    # net = slim.fully_connected(net, num_outputs=NUM_CLASSES,
    #                             activation_fn=None, scope='Predict')
    
# 获取预训练的模型的参数列表
variables_to_restore = slim.get_variables_to_restore(exclude=["resnet_v1_50/logits", "resnet_v1_50/predictions"])

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=net)

train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(net, axis=1))[1]

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merge_op = tf.summary.merge_all()


saver_restore = tf.train.Saver(var_list=variables_to_restore)
saver = tf.train.Saver(tf.global_variables())


sess = tf.Session()
sess.run([iterator.initializer, test_iterator.initializer, tf.global_variables_initializer(), tf.local_variables_initializer()])
# 导入预训练模型参数
saver_restore.restore(sess, "./models/resnet_v1_50.ckpt")

writer = tf.summary.FileWriter('./log', sess.graph)

index = 0
while True:
    try:
        img, label = sess.run([images, labels])
        _ = sess.run([train_op], feed_dict={
            tf_x: img, tf_y:label
        })
        if index % 10 == 0:
            result, _acc , _loss = sess.run([merge_op, accuracy, loss], feed_dict={
                tf_x: img, tf_y: label
            })
            writer.add_summary(result, index)
            print("%d step   |   %f accuarcy   | %f loss " % (index, _acc, _loss))
    except tf.errors.OutOfRangeError:
        break
    index += 1

sess.close()
