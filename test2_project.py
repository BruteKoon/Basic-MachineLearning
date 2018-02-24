import tensorflow as tf, sklearn
from librosa.feature import mfcc
import numpy as np
import random
import glob
import os.path
from scipy.io import wavfile

##########################setting ######################################
fs = 44100
n_mfcc = 13
# specify length of time slice in sec
time_slice = 5
data_per_time_slice = int(fs * time_slice)

# hyper parameters
data_set_length = 5603
test_batch_size = 1000
one_test = np.ones((test_batch_size, 1))
zero_test = np.zeros((test_batch_size, 1))
savepath = "sav/model.ckpt"
resultfile = 'result/result.txt'

total_accuracy_text = 0.0
total_a = 0
total_b = 0
total_c = 0

total_d = 0
total_e = 0
total_f = 0

total_g = 0
total_h = 0
total_i = 0
#######################################################################


print("importing and processing started")

# directory/file -> list

fileindex = 1

testpath = 'txt/*/*.txt'


#########################################################################
keep_prob = tf.placeholder(tf.float32)
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
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[100, 3], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([3]))
logits = tf.matmul(L3, W4) + b4
########################################################################3








filenames = glob.glob(testpath)
for filename in filenames:
    test_total_y_batch = np.array([], dtype=np.int64).reshape(0, 3)
    test_set = np.array([], dtype=np.int64).reshape(0, data_set_length)
    print(fileindex, ': ', filename)
    fileindex = fileindex + 1
    # "txt" -> array
    text_array = np.loadtxt(filename)
    text_array = np.transpose(text_array)
    text_array = text_array[2]
    ########### make y data set ###########################################
    if (filename.upper().find("UNLOADED") > -1):
        test_y_batch = np.hstack((zero_test, one_test, zero_test))
    elif (filename.upper().find("LOADED") > -1):
        test_y_batch = np.hstack((zero_test, zero_test, one_test))
    else:
        test_y_batch = np.hstack((one_test, zero_test, zero_test))

    test_total_y_batch = np.vstack((test_total_y_batch, test_y_batch))
    ############################################################################
    ###################for loop for test set##################################
    for i in range(test_batch_size):
        # select random number
        #print(text_array.shape[0])
        #print(data_per_time_slice)
        rand_num = random.randrange(0, text_array.shape[0] - data_per_time_slice)
        # make a random_array
        random_array = text_array[rand_num:rand_num + data_per_time_slice]

        pre_emphasis = 0.97
        emphasized_signal = np.append(random_array[0], random_array[1:] - pre_emphasis * random_array[:-1])

        mfccs = mfcc(y=emphasized_signal, sr=fs, S=None, n_mfcc=n_mfcc)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

        # print(S_norm.shape)
        # data_set.append(S_norm.reshape(-1))
        test_set = np.vstack((test_set, mfccs.reshape(-1)))

    with tf.Session() as sess:
    # initialize
        saver = tf.train.Saver()
        if os.path.isfile(savepath + ".index"):
            saver.restore(sess, savepath)
            print("model restored")
        else:
            sess.run(tf.global_variables_initializer())
            print("model generated")

        # Test model and check accuracy
        ##########################################################################

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        table_generator = tf.contrib.metrics.confusion_matrix(tf.reshape(tf.argmax(Y, 1), [-1]),
                                                          tf.reshape(tf.argmax(logits, 1), [-1]))
        feed_dict = {X: test_set, Y: test_total_y_batch, keep_prob: 1.0}
        testaccuracy = sess.run(accuracy, feed_dict=feed_dict)
        table = sess.run(table_generator, feed_dict=feed_dict)

        testsize = len(test_set)
 
        total_accuracy_text = testaccuracy + total_accuracy_text
        try:
            total_a = total_a + table[0][0]
        except IndexError:
            total_a = total_a + 0
        try:
            total_b = total_b + table[0][1]
        except IndexError:
            total_b = total_b + 0
        try:
            total_c = total_c + table[0][2]
        except IndexError:
            total_c = total_c + 0
        try:
            total_d = total_d + table[1][0]
        except IndexError:
            total_d = total_d + 0
        try:
            total_e = total_e + table[1][1]
        except IndexError:
            total_e = total_e + 0
        try:
            total_f = total_f + table[1][2]
        except IndexError:
            total_f = total_f + 0
        try:
            total_g = total_g + table[2][0]
        except IndexError:
            total_g = total_g + 0
        try:
            total_h = total_h + table[2][1]
        except IndexError:
            total_h = total_h + 0
        try:
            total_i = total_i + table[2][2]
        except IndexError:
            total_i = total_i + 0

    
    ###########################################################################
################################CNN MODEL ################################33	
    #print(accuracy_text)
    #print(table)
    #print(size_text)
    #print(bad_text)
print(total_accuracy_text/56)
print(str(total_a) + "    " + str(total_b) + "    " + str(total_c))
print(str(total_d) + "    " + str(total_e) + "    " + str(total_f))
print(str(total_g) + "    " + str(total_h) + "    " + str(total_i))
'''
              noise  unload  loaded     prediction
            +-------+-------+-------+
    noise   |       |       |       |           
            +-------+-------+-------+
    unload  |       |       |       |
            +-------+-------+-------+
    loaded  |       |       |       |
            +-------+-------+-------+
    answer

'''
