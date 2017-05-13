# neural style transfer
# vgg network
# gram metrix

import tensorflow as tf
import tensorflow.contrin.slim as slim

# def vgg_16(inputs, reuse=False, final_enpoint="fc8"):
#   inputs *= 255.0
  # inputs -= tf.contant([123.68, 1126.])

def main():
	g = tf.Graph()
	with g.as_default():
		img = tf.Variable()
		_, end_point = vgg_16(img)

		content_loss = get_content_loss(end_points)
		style_loss = get_style_loss(end_points)		
		loss = content_loss + style_loss

		opt = tf.train.GradientDescentOptimizer(1.0)
		train_op = opt.minimize(loss)

		with tf.Session() as sess:
			for steps in xrange(20000):
				sess.run(train_op)
