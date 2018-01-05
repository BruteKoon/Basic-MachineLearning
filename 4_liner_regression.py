import tensorflow as tf
#X,Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]
#Trainable Variable(It is a Variable that Tensorflow use)
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
#hypothesis X*W+B
hypothesis = x_train * w + b

#tf.redue_mean is a Average function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

# if you use the 'tf.Variable' then you must do this 'tf.global_~~~'
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(cost), sess.run(w), sess.run(b))

