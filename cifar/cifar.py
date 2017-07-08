import getopt
import sys
import os
import re
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cPickle

class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def get_data_label_old(dirs, training):
    NUM_IMS = 50000 if training else 10000
    FILE_NAME_PREFIX = r"data_batch_\d+" if training else r"test_batch"
    datas = np.zeros(shape=[NUM_IMS, 3072], dtype=np.float32)
    labels = []
    begin = 0
    for fi in os.listdir(dirs):
        if re.match(FILE_NAME_PREFIX, fi):
            d = unpickle(os.path.join(dirs, fi))
            end = begin + len(d['data'])
            datas[begin:end,:] = d['data']
            labels += d['labels']
            begin = end
    print "Loaded data size: %d" % len(datas)
    print "Loaded labels size: %d" % len(labels)
    return datas, labels

def get_data_label(dirs, training):
    filename = "data_batch_1"
    d = unpickle(os.path.join(dirs,filename))
    datas = d['data']
    labels = np.array(d['labels'])
    datas = datas.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    return datas, labels

def get_next_batch(datas, labels, size):
    selected = set()
    for _ in range(0, size):
        index = random.randint(0, len(datas)-1) 
        while index in selected:
           index += 1
           index %= len(datas)
        selected.add(index)

    s_datas =  []
    s_labels = []
    for index in selected:
        s_datas.append(datas[index])
        right_anwser = labels[index]
        label = [] 
        for _ in range(0, 10):
            if _ == right_anwser:
                label.append(1)
            else:
                label.append(0)
        s_labels.append(label)
    s_datas = np.concatenate([s_datas]).astype(np.float32)
    s_datas = s_datas.reshape([-1, 32, 32, 3])
    s_labels = np.concatenate([s_labels]).astype(np.int32)
    #print s_datas.shape
    #print s_labels.shape
    return s_datas, s_labels

#Constants
MODEL_SAVE_PATH = "./model.chkpt"
KEEP_PROB = 0.94
BATCH_SIZE = 10
ITER_NUM = 2000

def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')#% , data_format="NCHW")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                            strides=[1,2,2,1], padding='SAME')

## The input neural
#x = tf.placeholder(tf.float32, [None, 32*32*3])
#x_image = tf.reshape(x, [-1, 32, 32, 3])
x_image = tf.placeholder(tf.float32, [None, 32, 32, 3])

#the filter of conv
W_conv1 = weight_var([5,5,3,96])

#the bias of first conv
b_conv1 = bias_var([96])


h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1))
h_pool1 = max_pool_2x2(h_conv1)


## The second conv and max_pool
W_conv2 = weight_var([5, 5, 96, 96])
b_conv2 = bias_var([96])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## the dense connected layer
W_fc1 = weight_var([8*8*96, 1024])
b_fc1 = bias_var([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*96])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


##Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


##Readout layer
W_fc2 = weight_var([1024, 10])
b_fc2 = bias_var([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# the expected output, this comes from the label
y_ = tf.placeholder(tf.int32, [None, 10])

# the Error function of each output
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels = y_, logits = y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train_and_save(batch_size, iter_num):
    datas, labels = get_data_label("cifar-10-batches-py", training=True)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run(session=sess)

    tf.summary.merge_all()
    writer = tf.summary.FileWriter("./", sess.graph)
    saver = tf.train.Saver()
    tf.add_to_collection('train_op', train_step)
    # actual training procedure
    for i in range(iter_num):
        batch_xs, batch_ys = get_next_batch(datas, labels, batch_size)
        if i % (iter_num/10) == 0:
            #print_graph_info()
            print("accrucy ")
            print(sess.run(accuracy, feed_dict={x_image: np.array(batch_xs), y_: np.array(batch_ys), keep_prob:KEEP_PROB}))
        summary = sess.run(train_step, feed_dict={x_image: batch_xs, y_: batch_ys, keep_prob:KEEP_PROB})

    writer.add_summary(summary, i)

    saved_path = saver.save(sess, MODEL_SAVE_PATH)
    print ("Model saved in file:%s" % saved_path)


def load_and_apply():
    datas, labels = get_data_label("cifar-10-batches-py", training = False)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_SAVE_PATH)
    test_num = len(datas)
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    # Visualize the pixels using matplotlib
    def key_event(e):
        print e.key
        if e.key == 'right' or e.key == 'left':
            random_test_batch()

    def random_test_batch():
        batch_size = 16 
        NUM_HI = 4
        NUM_WID = 4
        batch_xs, batch_ys = get_next_batch(datas, labels, batch_size)
        classes = sess.run(tf.argmax(y_conv,1), feed_dict={x_image: batch_xs,
                                            y_: batch_ys, 
                                            keep_prob:KEEP_PROB})
        print ",".join(class_names[_] for _ in classes)
    
        for i in range(NUM_HI):
            for j in range(NUM_WID):
                axes[i][j].set_axis_off()
                axes[i][j].set_title(class_names[classes[i*NUM_HI+j]], fontsize=9)
                #axes[i][j].imshow(batch_xs[i*NUM_HI+j],interpolation='hanning')
                axes[i][j].imshow(batch_xs[i*NUM_HI+j])
        plt.show()
    fig.canvas.mpl_connect('key_press_event', key_event)
    random_test_batch() 

def plot_images(hirozontal, vertical):
    datas, labels = get_data_label("cifar-10-batches-py", training = False)
    fig, axes1 = plt.subplots(hirozontal, vertical,figsize=(4,4))
    for j in range(hirozontal):
        for k in range(vertical):
            i = np.random.choice(range(len(datas)))
            axes1[j][k].set_axis_off()
            axes1[j][k].set_title(class_names[labels[i:i+1][0]],fontsize=9)
            #axes1[j][k].imshow(datas[i:i+1][0])
            axes1[j][k].imshow(datas[i:i+1][0], interpolation='hanning')
    plt.show()


def usage():
    sys.stderr.write("Usage: python %s [-t | --train] [-b | --batch] [--apply | -a] [-h] \n\n" % __file__)

def print_graph_info():
   # print x.get_shape()
    print x_image.get_shape()
    print y_conv.__class__.__name__
    print y_.__class__.__name__
    print "y_conv %s " % str( y_conv.get_shape())
    print "y_ %s " % str( y_.get_shape())
    print "h_pool1 %s" % str(h_pool1.get_shape())
    print "h_pool2 %s" % str(h_pool2.get_shape())
    print "h_fc1 %s" % str(h_fc1.get_shape())


if __name__ == "__main__":

    #get options
    opts, args = getopt.getopt(sys.argv[1:], 'phtb:ai:', ['train','batch=', 'apply', 'iter='])
    is_training = False
    batch_size = BATCH_SIZE
    iter_num = ITER_NUM 
    is_apply = False
    isplot = False
    for opt, value in opts:
        if opt =='-h':
            usage()
            sys.exit(0)
        if opt =='--train' or opt=='-t':
            is_training = True
        if opt == '--batch' or opt=='-b':
            batch_size = int(value)
        if opt == '--iter':
            iter_num = int(value)
        if opt =='--apply' or opt=='-a':
            is_apply = True
        if opt == "-p":
            isplot = True
    # check options
    if is_training and is_apply:
        usage()

    # action based on option
    if is_training:
        train_and_save(batch_size, iter_num)
    elif is_apply:
        load_and_apply()
    elif isplot:
        plot_images(3,3)
    else:
        print_graph_info()
