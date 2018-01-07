import tensorflow as tf

#tf Graph Input
X = [1,2,3]
Y = [1,2,3]

#Set wronig model weights
W = tf.Variable(-3.0)

#Linear model
hypothesis = X * W
#cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize: gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(100):
	print(step, sess.run(W))
	sess.run(train)
