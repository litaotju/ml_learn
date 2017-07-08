import getopt
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
# define the architecture is simple
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Constants
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
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                            strides=[1,2,2,1], padding='SAME')

## The input neural
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

#the filter of conv
W_conv1 = weight_var([5,5,1,32])

#the bias of first conv
b_conv1 = bias_var([32])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


## The second conv and max_pool
W_conv2 = weight_var([5, 5, 32, 64])
b_conv2 = bias_var([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## the dense connected layer
W_fc1 = weight_var([7*7*64, 1024])
b_fc1 = bias_var([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


##Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


##Readout layer
W_fc2 = weight_var([1024, 10])
b_fc2 = bias_var([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# the expected output, this comes from the label
y_ = tf.placeholder(tf.float32, [None, 10])

# the Error function of each output
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels = y_, logits = y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train_and_save(batch_size, iter_num):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run(session=sess)

    tf.summary.merge_all()
    writer = tf.summary.FileWriter("./mnist_softmax_summary", sess.graph)
    saver = tf.train.Saver()
    tf.add_to_collection('train_op', train_step)
    # actual training procedure
    for i in range(iter_num):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        if i % (iter_num/10) == 0:
            print("accrucy ")
            print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:KEEP_PROB}))
        summary = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:KEEP_PROB})

    writer.add_summary(summary, i)

    saved_path = saver.save(sess, MODEL_SAVE_PATH)
    print ("Model saved in file:%s" % saved_path)


def load_and_apply(input_pic):
    print("input picture %s" % input_pic) 

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_SAVE_PATH)

    fig = plt.figure()

    # Visualize the pixels using matplotlib
    def key_event(e):
        print e.key
        if e.key == 'right' or e.key == 'left':
            Nth = random.randint(0, len(mnist.test.images)-2)
            test_Nth(Nth) 

    def test_Nth(Nth):
        print(sess.run(y_conv, feed_dict={x: mnist.test.images[Nth: Nth+1],
                                            y_: mnist.test.labels[Nth: Nth+1],
                                            keep_prob:KEEP_PROB}))
        pixels = np.array(mnist.test.images[Nth]).reshape((28,28))
        label = np.argmax(np.array(mnist.test.labels[Nth]), axis=0)
        plt.title('Label is {}'.format(label))
        plt.imshow(pixels, cmap='gray')
        plt.show()

    fig.canvas.mpl_connect('key_press_event', key_event)
    Nth = random.randint(0, len(mnist.test.images)-2)
    test_Nth(Nth) 
    
def usage():
    sys.stderr.write("Usage: python %s [-t | --train] [-b | --batch] [--apply | -a] [--input | -i ='input picture path'][-h] \n\n" % __file__)

if __name__ == "__main__":

    #get options
    opts, args = getopt.getopt(sys.argv[1:], 'htb:ai:', ['train','batch=', 'apply', 'input=', 'iter='])
    is_training = False
    batch_size = BATCH_SIZE
    iter_num = ITER_NUM 
    is_apply = False
    input_pic = ""
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
        if opt =="--input" or opt=='-i':
            input_pic = value 

    # check options
    if is_training and is_apply:
        usage()
    if is_apply and input_pic == "":
        usage()

    # action based on option
    if is_training:
        train_and_save(batch_size, iter_num)
    if is_apply:
        load_and_apply(input_pic)

