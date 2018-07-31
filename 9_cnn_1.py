""" import your model here """
import numpy as np
np.set_printoptions(precision = 1, linewidth = 180, threshold = np.inf)
import tensorlow as tf
""" your model should support the following code """


def weight_variable(shape):
	initial = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME')

# input
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool1 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool1, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Get the mnist dataset (use tensorflow here)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# train and eval
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# train_accuracy = sess.run([grad1, grad2, cross_entropy], feed_dict = {x: mnist.train.images[:50], y_: mnist.train.labels[:50]})
	# print('Step %d, trainning accuracy %s' % (-1, str(train_accuracy[0])))
	# print(train_accuracy[1])
	# print(train_accuracy[2])
	# exit(0)
	for i in range(200):
		batch = mnist.train.next_batch(50)
		if i % 3 == 0:
			train_accuracy = sess.run([accuracy, cross_entropy], feed_dict = { x: batch[0], y_: batch[1]})
			print('Step %d, trainning accuracy %s' % (i, str(train_accuracy)))
			# print(train_accuracy[0][0])
			# print(train_accuracy[0][1])
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})

	print("trainFinished")
	ans = accuracy.eval(feed_dict={ x:mnist.test.images,
									y_: mnist.test.labels})
	print('Test accuracy: %g' % ans)
	assert ans > 0.88


















# """ import your model here """
# import tensorlow as tf
# import numpy as np
# np.set_printoptions(precision = 1, linewidth = 180, threshold = np.inf)
#
#
# """ your model should support the following code """
#
#
# # img = [[[[1], [3], [8], [0], [3]],
# #		 [[0], [1], [5], [1], [0]],
# #		 [[9], [0], [1], [7], [1]],
# #		 [[0], [8], [1], [4], [0]],
# #		 [[8], [1], [1], [0], [6]]]]*1
# #
# # flt = np.array([ [ [[1] * 1], [[0] * 1], [[1] * 1] ],
# #		 [ [[0] * 1], [[1] * 1], [[0] * 1] ],
# #		 [ [[1] * 1], [[0] * 1], [[1] * 1] ]])
# # print(np.shape(flt))
# # fflt = flt[0:2,0:2]
# # print(img)
# # print(np.reshape(flt, (3, 3)))
# # image = tf.Variable(img, dtype = tf.float64)
# # filter = tf.Variable(fflt, dtype = tf.float64)
# # conv = tf.nn.conv2d(image, filter, strides = [1, 1, 1, 1], padding = "SAME")
# # # res = tf.nn.max_pool(conv, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
# # # grad = tf.gradients(res, [image, filter])
# # with tf.Session() as sess:
# # 	sess.run((tf.global_variables_initializer()))
# # 	print(sess.run(tf.reshape(conv, (5, 5))), end = "\n\n")
# # 	# print(sess.run(res), end="\n\n")
# # 	#
# # 	# tmp = sess.run(grad)
# # 	# print(tmp[0], end = "\n\n")
# # 	# print(tmp[1])
# # exit(0)
#
# def weight_variable(shape):
# 	# initial = [ [[[1]], [[3]], [[8]], [[0]], [[3]]],
# 	# 			[[[0]], [[1]], [[5]], [[1]], [[0]]],
# 	# 			[[[9]], [[0]], [[1]], [[7]], [[1]]],
# 	# 			[[[0]], [[8]], [[1]], [[4]], [[0]]],
# 	# 			[[[8]], [[1]], [[1]], [[0]], [[6]]]]
# 	initial = tf.random_normal(shape)
# 	return tf.Variable(initial, name = "W", dtype = tf.float32)
#
# def bias_variable(shape):
# 	initial = tf.constant(0.1, shape=shape)
# 	return tf.Variable(initial, name = "b")
#
# def conv2d(x, W):
# 	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#
# def max_pool_2x2(x):
# 	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
# 			padding='SAME')
#
# # input
# x = tf.placeholder(tf.float32, shape=[None, 784], name = "x")
# y_ = tf.placeholder(tf.float32, shape=[None, 10], name = "y_")
#
# # first layer
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # second layer
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # densely connected layer
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool1, [-1, 7 * 7 * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# # readout layer
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
#
# # loss
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#
# train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # Get the mnist dataset (use tensorflow here)
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
#
# # train and eval
# with tf.Session() as sess:
# 	# sess.run(tf.global_variables_initializer())
# 	# print(sess.run(tf.reshape(x_image, (28, 28)), feed_dict = {x: mnist.train.images[0:1], y_: mnist.train.labels[0:1]}))
# 	# print(sess.run(tf.reshape(W_conv1, (5, 5)), feed_dict = {x: mnist.train.images[0:1], y_: mnist.train.labels[0:1]}))
# 	# tmp = sess.run(tf.reshape(conv2d(x_image, W_conv1), (24, 24)), feed_dict = {x: mnist.train.images[0:1], y_: mnist.train.labels[0:1]})
# 	# print(tmp)
# 	#
# 	# exit(0)
# 	for i in range(30):
# 		batch = mnist.train.next_batch(50)
# 		if (i + 1) % 2 == 0:
# 			train_accuracy = accuracy.eval(feed_dict = { x: batch[0], y_: batch[1]})
# 			print('Step %d, trainning accuracy %g' % (i, train_accuracy))
#
# 		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
# 	print("train finished")
# 	ans = accuracy.eval(feed_dict={ x:mnist.test.images,
# 									y_: mnist.test.labels})
# 	print('Test accuracy: %g' % ans)
# 	assert ans > 0.88
