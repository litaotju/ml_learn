from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# define the architecture is simple
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# this is enough to describe the arch
y = tf.nn.softmax(tf.matmul(x, W)+b)

# the expected output, this comes from the label
y_ = tf.placeholder(tf.float32, [None, 10])

# the Error function of each output
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#    labels = y_, logits = y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.summary.merge_all()
writer = tf.summary.FileWriter("./simple_softmax_mnist", sess.graph)

tf.global_variables_initializer().run(session=sess)


# actual training procedure
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    writer.add_summary(summary, i)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(sess.run([W,b]))
