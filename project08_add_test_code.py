import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import random
import glob

##########################setting ######################################
fs = 44100

# specify length of time slice in sec
time_slice = 5
data_per_time_slice = int(fs * time_slice)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 10
one = np.ones((batch_size, 1))
zero = np.zeros((batch_size, 1))
total_y_batch = np.array([], dtype=np.int64).reshape(0,3)
data_set = np.array([], dtype=np.int64).reshape(0,441856)
#######################################################################



print("importing and processing started")

# directory/file -> list
path = 'txt/*.txt'
filenames = glob.glob(path)   
for filename in filenames:
    print(filename)
    # "txt" -> array
    text_array = np.loadtxt(filename)
    text_array = np.transpose(text_array)
############ make y data set ###########################################
    if(filename.upper().find("UNLOADED")>-1):
        y_batch = np.hstack((zero, one, zero))
    elif(filename.upper().find("LOADED")>-1):
        y_batch = np.hstack((zero, zero, one))
    else:
        y_batch = np.hstack((one, zero, zero))
   
    total_y_batch = np.vstack((total_y_batch, y_batch))
############################################################################

####################### make x data set ################################
# for loop ################################################################
    for i in range(batch_size):
        # select random number
        # print(data_per_time_slice)
        rand_num = random.randrange(0, text_array[1].shape[0] - data_per_time_slice)
        # make a random_array
        random_array = text_array[2][rand_num:rand_num + data_per_time_slice]

        # view graph
        f, t, S = signal.stft(random_array, fs, window='hamm', nperseg=512, nfft=1023)
        S = np.abs(S)
        S = 20 * np.log10(S + 1e-6)

        maximum = max(max(S.reshape([1, -1])))
        minimum = min(min(S.reshape([1, -1])))
        S_norm = (S - minimum) / (maximum - minimum)
        
        # print(S_norm.shape)
        # data_set.append(S_norm.reshape(-1))
        data_set = np.vstack((data_set, S_norm.reshape(-1)))
###############################################################################
print("importing and processing done")

#################################CNN MODEL ################################33
# input place holders
X = tf.placeholder(tf.float32, [None, 441856])
X_img = tf.reshape(X, [-1, 512, 863, 1])   # img 512x863x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 3])

# L1 ImgIn shape=(?, 512, 863, 1)
W1 = tf.Variable(tf.random_normal([10, 10, 1, 32], stddev=0.01))
#    Conv     -> (?, 512, 863, 32)
#    Pool     -> (?, 128, 216, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1], padding='SAME')

# L2 ImgIn shape=(?, 128, 216, 32)
W2 = tf.Variable(tf.random_normal([10, 10, 32, 64], stddev=0.01))
#    Conv      ->(?, 128, 216, 64)
#    Pool      ->(?, 64, 54, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 4, 1],strides=[1, 2, 4, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 64*54*64])

# Final FC 64x54x64 inputs -> 3 outputs
W3 = tf.get_variable("W3", shape=[64 * 54 * 64, 3],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([3]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
##############################################################################

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(data_set.shape[0]/batch_size)

    for i in range(total_batch):
        feed_dict = {X: data_set[i*batch_size:(i+1)*batch_size], Y: total_y_batch[i*batch_size:(i+1)*batch_size]}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        print(c)
        avg_cost += c/total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: data_set, Y: total_y_batch}))
