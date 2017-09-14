from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import xlrd

# step 1: read data
book = xlrd.open_workbook('data/slr05.xls', encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data  =  np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# step 2: place holder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# step 3: weights and bias
w = tf.Variable(0.0, dtype=tf.float32, name='weights_1')
u = tf.Variable(0.0, dtype=tf.float32, name='weights_2')
b = tf.Variable(0.0, dtype=tf.float32, name='bias')

# step 4: model
Y_ = X*X*w + X*u + b

# step 5: loss function using square error
loss = tf.square(Y - Y_, name='loss')

# step 6: gradient descent
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

with tf.Session() as sess:
  # step 7: initialize variable
  sess.run(tf.global_variables_initializer())

  writer = tf.summary.FileWriter('./graphs/02_linear_reg', sess.graph)

  # step 8: train model
  for i in range(100):
    for x, y in data:
      sess.run(optimize, feed_dict={X: x, Y: y})

  w_value, u_value, b_value = sess.run([w, u, b])
  print([w_value, u_value, b_value])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X*X*w_value + X*u_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
