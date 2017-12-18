import numpy as np
import tensorflow as tf
import cv2, glob, sys

n_images = 2057 + 1364
n_classes = 2
height, width, channels = 50, 50, 3
learning_rate = 0.000001

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

	sess.run(init)

	saver.restore(sess, 'model/eyerecog4.ckpt')

	test_images = glob.glob('../images/test/*.*')
	
	for path_test in test_images:

		img_test = cv2.imread(path_test)
		cv2.imshow('img', img_test)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	
		img_test_copy = img_test
		img_test = img_test / 255
		h,w,_ = img_test.shape
		stride_h = h // 6
		stride_w = w // 6
		threshold = 0.995

		for i in range(0, h-1, stride_h // 6):

			for j in range(0, w-1, stride_w // 6):

				img_box = cv2.resize(img_test[i:i+stride_h, j:j+stride_w], (width, height), interpolation = cv2.INTER_AREA)
				box_prediction = sess.run(prediction, feed_dict={X: img_box.reshape(-1, width, height, channels)})
				if(box_prediction[0][0] > threshold and box_prediction[0][0] < 0.999 ):
					# print(box_prediction)
					for k in range(i,i+stride_h,1):
						for l in range(j,j+stride_w,1):
							if (k < h) and (l < w):
								if img_test_copy[k,l][2] > img_test_copy[k,l][0]*2 and img_test_copy[k,l][2] > img_test_copy[k,l][1]*2:
									img_test_copy[k,l][2] = 30
	
		cv2.imshow('img', img_test_copy)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	