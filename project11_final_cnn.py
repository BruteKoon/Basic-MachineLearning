import tensorflow as tf, sklearn
from librosa.feature import mfcc
import numpy as np
import random
import glob

##########################setting ######################################
fs = 44100
n_mfcc = 13
# specify length of time slice in sec
time_slice = 5
data_per_time_slice = int(fs * time_slice)

# hyper parameters
data_set_length = 5603
learning_rate = 0.001
training_epochs = 8
batch_size = 10
test_batch_size = 20
one = np.ones((batch_size, 1))
zero = np.zeros((batch_size, 1))
one_test = np.ones((test_batch_size, 1))
zero_test = np.zeros((test_batch_size, 1))
total_y_batch = np.array([], dtype=np.int64).reshape(0, 3)
test_total_y_batch = np.array([], dtype=np.int64).reshape(0, 3)
data_set = np.array([], dtype=np.int64).reshape(0, data_set_length)
test_set = np.array([], dtype=np.int64).reshape(0, data_set_length)
gain = 1.0
savepath = "sav/model2.ckpt"
#######################################################################


print("importing and processing started")

# directory/file -> list
path = 'txt/training_sample/*.txt'
filenames = glob.glob(path)
print("training set start")
for filename in filenames:
    print(filename)
    # "txt" -> array
    training_text_array = np.loadtxt(filename)
    training_text_array = np.transpose(training_text_array)
    training_text_array = training_text_array[2]
    ########### make y data set ###########################################
    if (filename.upper().find("UNLOADED") > -1):
        y_batch = np.hstack((zero, one, zero))
    elif (filename.upper().find("LOADED") > -1):
        y_batch = np.hstack((zero, zero, one))
    else:
        y_batch = np.hstack((one, zero, zero))
    total_y_batch = np.vstack((total_y_batch, y_batch))
    ###########################################################################
    ########################## make x data set ################################
    # for loop for data_set################################################################
    for i in range(batch_size):
        # select random number
        # print(data_per_time_slice)
        rand_num = random.randrange(0, training_text_array.shape[0] - data_per_time_slice)
        # make a random_array
        random_array = training_text_array[rand_num:rand_num + data_per_time_slice]

        pre_emphasis = 0.97
        emphasized_signal = np.append(random_array[0], random_array[1:] - pre_emphasis * random_array[:-1])

        mfccs = mfcc(y=emphasized_signal, sr=fs, S=None, n_mfcc=n_mfcc)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        # print(S_norm.shape)
        # data_set.append(S_norm.reshape(-1))
        data_set = np.vstack((data_set, mfccs.reshape(-1)))
    ############################################################################
    ###########################################################################
print("training_Set done");
print("test_set start");

path = 'txt/test_sample/*.txt'
test_filenames = glob.glob(path)
for filename in test_filenames:
    print(filename)
# "txt" -> array
    test_text_array = np.loadtxt(filename)
    test_text_array = np.transpose(test_text_array)
    test_text_array = test_text_array[2]
	
    ########### make y data set ###########################################
    if (filename.upper().find("UNLOADED") > -1):
        test_y_batch = np.hstack((zero_test, one_test, zero_test))
    elif (filename.upper().find("LOADED") > -1):
        test_y_batch = np.hstack((zero_test, zero_test, one_test))
    else:
        test_y_batch = np.hstack((one_test, zero_test, zero_test))
    test_total_y_batch = np.vstack((test_total_y_batch, test_y_batch))
    ###################for loop for test set##################################
    for i in range(test_batch_size):
        # select random number
        # print(data_per_time_slice)
        rand_num = random.randrange(0, test_text_array.shape[0] - data_per_time_slice)
        # make a random_array
        random_array = test_text_array[rand_num:rand_num + data_per_time_slice]

        pre_emphasis = 0.97
        emphasized_signal = np.append(random_array[0], random_array[1:] - pre_emphasis * random_array[:-1])

        mfccs = mfcc(y=emphasized_signal, sr=fs, S=None, n_mfcc=n_mfcc)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

        # print(S_norm.shape)
        # data_set.append(S_norm.reshape(-1))
        test_set = np.vstack((test_set, mfccs.reshape(-1)))

print("test_Set done")
print("importing and processing done")

################################CNN MODEL ################################33
# input place holders
X = tf.placeholder(tf.float32, [None, data_set_length])
X_img = tf.reshape(X, [-1, 13, 431, 1])  # img 13x431x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 3])
# L1 ImgIn shape=(?, 13, 431, 1)
W1 = tf.Variable(tf.random_normal([2, 4, 1, 32], stddev=0.01))
#    Conv     -> (?, 13, 431, 32)
#    Pool     -> (?, 7, 108, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')

# L2 ImgIn shape=(?, 7, 108, 32)
W2 = tf.Variable(tf.random_normal([2, 9, 32, 64], stddev=0.01))
#    Conv      ->(?, 7, 108, 64)
#    Pool      ->(?, 7, 54, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7 * 54 * 64])

# Final FC 7x54x64 inputs -> 100 outputs
W3 = tf.get_variable("W3", shape=[7 * 54 * 64, 100], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([100]))
L3 = tf.matmul(L2_flat, W3) + b3
L3 = tf.nn.relu(L3)

W4 = tf.get_variable("W4", shape=[100, 3], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([3]))
logits = tf.matmul(L3, W4) + b4

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#############################################################################

with tf.Session() as sess:
    # initialize
    saver = tf.train.Saver()
    # sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, savepath)
    print("model restored")


    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(data_set.shape[0] / batch_size)

        for i in range(total_batch):
            feed_dict = {X: data_set[i * batch_size:(i + 1) * batch_size],
                         Y: total_y_batch[i * batch_size:(i + 1) * batch_size]}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            print(c)
            avg_cost += c / total_batch
        save_path = saver.save(sess, savepath)
        print("Model saved in path: %s" % save_path)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    ##########################################################################

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: test_set, Y: test_total_y_batch}))
