import tensorflow as tf
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# HYPOTHESIS XW+b
hypothesis = X * W + b

#Average(cost function)
cost = tf.reduce_mean(tf.square(hypothesis-Y))

#Minimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a Session
sess = tf.Session()

# Initailizer global variables in the graph
sess.run(tf.global_variables_initializer())

# FIt the LIne with new training data
for step in range(5001):
	cost_val, W_val, b_val, _ = \
		sess.run([cost, W,  b, train],
			feed_dict={X: [1,2,3], Y:[1,2,3]})
	if step % 20 == 0:
		print(step, cost_val,  W_val, b_val)
