import tensorflow as tf

x_data = [[73,80,75], [93,88,93], [89,91,90], [96,98,100],[73,66,70]]
y_data = [[152], [185], [180], [196], [142]]

#placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Hypothesis
hypothesis = tf.matmul(X, w) + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
	cost_Val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
	if step % 10 == 0:
		print(step, "COst ", cost_Val, "\nPredictoin:\n", hy_val)
