import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

#MAke sure the shape and data are ok
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

#placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Hypothesis
hypothesis = tf.matmul(X,W) + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis- Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
#Initializes global variabls in the graph
sess.run(tf.global_variables_initializer())
#Sep up feed_dict variables inside the loop
for step in range(2001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X: x_data, Y:y_data})
	if step % 10 == 0:
		print(step, "COst: ", cost_val, "\nprediction:\n", hy_val)

#Ask my score
print("your score will be ", sess.run(hypothesis, feed_dict = {X: [[100,70,101]]}))
print("other scores will be ", sess.run(hypothesis, feed_dict = {X: [[60, 70, 110], [90, 100, 80]]}))

