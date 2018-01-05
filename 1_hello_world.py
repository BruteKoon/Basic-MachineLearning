import tensorflow as tf

# this op is added as a node to the default graph
hello = tf.constant("HEllo, TensorFlow!")

# seart a TF session
sess = tf.Session()

#run the op and get result
print(sess.run(hello))

