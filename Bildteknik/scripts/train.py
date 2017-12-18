import numpy as np
import tensorflow as tf
import cv2, glob, sys
from sklearn.utils import shuffle

n_images = 2057 + 1362 - 491
n_classes = 2
height, width, channels = 50, 50, 3

epochs = 10
batch_size = 1
n_batches = n_images // batch_size
learning_rate = 0.000001
i = 0


def get_data():

	eyes_dir = glob.glob('../images/eyes/*.*')
	non_eyes_dir = glob.glob('../images/non-eyes/*.*')

	x = np.ndarray((n_images, width, height, channels))
	y = np.ndarray((n_images, n_classes))
	global i

	for path in eyes_dir:
		img = cv2.imread(path)
		img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
		x[i] = img / 255
		y[i,0] = 1
		i += 1

	for path in non_eyes_dir:
		img = cv2.imread(path)
		img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
		x[i] = img / 255
		y[i,1] = 1
		i += 1

	return x, y

X = tf.placeholder(tf.float32, shape=[None, width, height, channels])
Y = tf.placeholder(tf.float32, shape=[None, n_classes])

conv1 = tf.layers.conv2d(X, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
max1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, padding="VALID")
conv3 = tf.layers.conv2d(max1, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
conv4 = tf.layers.conv2d(conv3, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
max2 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2, padding="VALID")
conv5 = tf.layers.conv2d(max2, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
conv6 = tf.layers.conv2d(conv5, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
max3 = tf.layers.max_pooling2d(conv6, pool_size=[2, 2], strides=2, padding="VALID")
conv7 = tf.layers.conv2d(max3, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
conv8 = tf.layers.conv2d(conv7, filters=16, kernel_size=3, activation=tf.nn.relu, strides=[1, 1], padding="SAME")
max4 = tf.layers.max_pooling2d(conv8, pool_size=[2, 2], strides=2, padding="VALID")
# print(max4.shape)
max3_flat = tf.reshape(max4, [-1, 3*3*16])
dense1 = tf.layers.dense(max3_flat, units=1000, activation=tf.nn.relu)
dense2 = tf.layers.dense(dense1, units=200, activation=tf.nn.relu)
logits = tf.layers.dense(dense2, units=n_classes)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	
	x, y = get_data()
	x,y = shuffle(x,y, random_state=42)
	y = y.astype(int)
	temp = np.asarray(y)
	zeros_indices = y < 0 
	y[zeros_indices] = 0
	test_size = i // 5

	x_test = x[0:test_size] 
	x = x[test_size:]
	y_test = y[0:test_size] 
	y = y[test_size:]	

	sess.run(init)
	saver.restore(sess, 'model/eyerecog4.ckpt')

	for epoch in range(1, epochs+1):

		for batch in range(n_batches):

			x_batch = x[batch*batch_size:(batch+1)*batch_size]
			y_batch = y[batch*batch_size:(batch+1)*batch_size]

			sess.run([train_op], feed_dict={X: x_batch, Y: y_batch})
		
		if epoch % 1 == 0:

			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x,
																 Y: y})
			loss_test, acc_test = sess.run([loss_op, accuracy], feed_dict={X: x_test,
																 Y: y_test})
			print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
			      "{:.4f}".format(loss) + ", Training Accuracy= " + \
			      "{:.3f}".format(acc) + \
			     	", Test Accuracy= " + "{:.3f}".format(acc_test))


	saver.save(sess, 'model/eyerecog4.ckpt')