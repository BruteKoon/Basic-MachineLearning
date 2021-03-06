import tensorflow as tf, sklearn
from librosa.feature import mfcc
import numpy as np
import random
import glob
import os.path

##########################setting ######################################
fs = 44100
n_mfcc = 13
# specify length of time slice in sec
time_slice = 5
data_per_time_slice = int(fs * time_slice)

# hyper parameters
data_set_length = 5603
learning_rate = 0.0001
training_epochs = 10
batch_size = 20
test_batch_size = 10
one = np.ones((batch_size, 1))
zero = np.zeros((batch_size, 1))
one_test = np.ones((test_batch_size, 1))
zero_test = np.zeros((test_batch_size, 1))
total_y_batch = np.array([], dtype=np.int64).reshape(0, 3)
test_total_y_batch = np.array([], dtype=np.int64).reshape(0, 3)
data_set = np.array([], dtype=np.int64).reshape(0, data_set_length)
test_set = np.array([], dtype=np.int64).reshape(0, data_set_length)
gain = 1.0
savepath = "sav/model.ckpt"
resultfile = 'result/result.txt'
#######################################################################


print("importing and processing started")

# directory/file -> list
path = 'txt/training_sample/*.txt'
testpath = 'txt/test_sample/*.txt'
filenames = glob.glob(path)

fileindex = 1

for filename in filenames:
    print(fileindex, ': ', filename)
    fileindex = fileindex + 1
    # "txt" -> array
    text_array = np.loadtxt(filename)
    text_array = np.transpose(text_array)
    text_array = text_array[2]
    ########### make y data set ###########################################
    if (filename.upper().find("UNLOADED") > -1):
        y_batch = np.hstack((zero, one, zero))
        test_y_batch = np.hstack((zero_test, one_test, zero_test))
    elif (filename.upper().find("LOADED") > -1):
        y_batch = np.hstack((zero, zero, one))
        test_y_batch = np.hstack((zero_test, zero_test, one_test))
    else:
        y_batch = np.hstack((one, zero, zero))
        test_y_batch = np.hstack((one_test, zero_test, zero_test))

    test_total_y_batch = np.vstack((test_total_y_batch, test_y_batch))
    total_y_batch = np.vstack((total_y_batch, y_batch))
    ###########################################################################
    ########################## make x data set ################################
    # for loop for data_set################################################################
    for i in range(batch_size):
        # select random number
        # print(data_per_time_slice)
        rand_num = random.randrange(0, text_array.shape[0] - data_per_time_slice)
        # make a random_array
        random_array = text_array[rand_num:rand_num + data_per_time_slice]

        pre_emphasis = 0.97
        emphasized_signal = np.append(random_array[0], random_array[1:] - pre_emphasis * random_array[:-1])

        mfccs = mfcc(y=emphasized_signal, sr=fs, S=None, n_mfcc=n_mfcc)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        # print(S_norm.shape)
        # data_set.append(S_norm.reshape(-1))
        data_set = np.vstack((data_set, mfccs.reshape(-1)))
    ############################################################################
    ###################for loop for test set##################################
    for i in range(test_batch_size):
        # select random number
        # print(data_per_time_slice)
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

    ###########################################################################

filenames = glob.glob(testpath)
for filename in filenames:
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
        # print(data_per_time_slice)
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

    ###########################################################################
print("importing and processing done")

################################CNN MODEL ################################33

keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, data_set_length])
X_img = tf.reshape(X, [-1, 13, 431, 1])  # img 13x431x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 3])
# L1 ImgIn shape=(?, 13, 431, 1)
with tf.name_scope("section1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 4, 1, 32], stddev=0.01), name='weight1')
    #    Conv     -> (?, 13, 431, 32)
    #    Pool     -> (?, 7, 108, 32)
    W1_hist = tf.summary.histogram("weights1", W1)

    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME', name = 'conv1')
    conv_hist = tf.summary.histogram("conv1",L1)

    L1 = tf.nn.relu(L1, name='relu1')
    relu_hist = tf.summary.histogram("relu1",L1)

    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME',name='pool1')
    pool_hist = tf.summary.histogram("pool1",L1)
    

with tf.name_scope("section2") as scope:
    # L2 ImgIn shape=(?, 7, 108, 32)
    W2 = tf.Variable(tf.random_normal([2, 9, 32, 64], stddev=0.01), name='weight2')
    W2_hist = tf.summary.histogram("weights2", W2)
    #    Conv      ->(?, 7, 108, 64)
    #    Pool      ->(?, 7, 54, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
    conv2_hist = tf.summary.histogram("conv2",L2)

    L2 = tf.nn.relu(L2, name='relu2')
    relu2_hist = tf.summary.histogram("relu2",L2)

    L2 = tf.nn.max_pool(L2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME', name='pool2')
    pool2_hist = tf.summary.histogram("pool2",L2)

    L2_flat = tf.reshape(L2, [-1, 7 * 54 * 64])
    

with tf.name_scope("section3") as scope:
    # Final FC 7x54x64 inputs -> 100 outputs
    W3 = tf.get_variable("W3", shape=[7 * 54 * 64, 100], initializer=tf.contrib.layers.xavier_initializer())
    W3_hist = tf.summary.histogram("weights3", W3)

    b3 = tf.Variable(tf.random_normal([100]), name='biases3')
    b3_hist = tf.summary.histogram("biases3", b3)

    L3 = tf.matmul(L2_flat, W3) + b3
    matmul3_hist = tf.summary.histogram("matmul3",L3)

    L3 = tf.nn.relu(L3, name='relu3')
    relu3_hist = tf.summary.histogram("relu3",L3)

    L3 = tf.nn.dropout(L3, keep_prob=keep_prob, name='dropout3')
    dropout3_hist = tf.summary.histogram("dropout3",L3)

with tf.name_scope("section4") as scope:
    W4 = tf.get_variable("W4", shape=[100, 3], initializer=tf.contrib.layers.xavier_initializer())
    W4_hist = tf.summary.histogram("weights4", W4)

    b4 = tf.Variable(tf.random_normal([3]), name='biases4')
    b4_hist = tf.summary.histogram('biases4', b4)

    logits = tf.matmul(L3, W4) + b4
    logits_hist = tf.summary.histogram("logits",logits)



# define cost/loss & optimizer
cost_weight = tf.nn.relu(tf.argmax(Y, 1) - tf.argmax(logits, 1)) + 1
_cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(tf.multiply(tf.cast(_cost, dtype=tf.float32), tf.cast(cost_weight, dtype=tf.float32)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#############################################################################
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy",accuracy)
cost_summ = tf.summary.scalar("cost", cost)
summary = tf.summary.merge_all()
global_step = 0

with tf.Session() as sess:
    # initialize
    saver = tf.train.Saver()
    if os.path.isfile(savepath+".index"):
        saver.restore(sess, savepath)
        print("model restored")
    else:
        sess.run(tf.global_variables_initializer())
        print("model generated")

    print('Learning started. It takes sometime.')
    writer = tf.summary.FileWriter("./logs",graph=sess.graph)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(data_set.shape[0] / batch_size)

        for i in range(total_batch):
            feed_dict = {X: data_set[i * batch_size:(i + 1) * batch_size],
                         Y: total_y_batch[i * batch_size:(i + 1) * batch_size],
                         keep_prob: 0.7}
            c,s ,_ = sess.run([cost, summary,optimizer], feed_dict=feed_dict)
            writer.add_summary(s, global_step=global_step)
            global_step += 1
            #print(c)
            avg_cost += c / total_batch
        save_path = saver.save(sess, savepath)
        print("Model saved in path: %s" % save_path)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')
    avg_cost_summ = tf.summary.scalar("avg_cost", avg_cost)
    # Test model and check accuracy
    ##########################################################################

    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    table_generator = tf.contrib.metrics.confusion_matrix(tf.reshape(tf.argmax(Y, 1), [-1]), tf.reshape(tf.argmax(logits, 1), [-1]))
    feed_dict = {X: test_set, Y: test_total_y_batch, keep_prob: 1.0}
    testaccuracy = sess.run(accuracy, feed_dict=feed_dict)
    table = sess.run(table_generator, feed_dict=feed_dict)

    testsize = len(test_set)
    badresult = table[1][0] + table[2][0] + table[2][1]

    accuracy_text = 'Accuracy: ' + str(testaccuracy) + '\n'
    size_text = '\nTest size: ' + str(testsize) + '\n'
    bad_text = 'Bad result: ' + str(badresult) + ' (' + str(badresult / testsize) + '%)\n\n'

    print(accuracy_text)
    print(table)
    print(size_text)
    print(bad_text)
    writer = tf.summary.FileWriter("./logs")
    #########################################################################
    ######################## write result to txt file #######################
    #########################################################################
    f = open(resultfile, 'a')
    # result_list = [accuracy_text, size_text, bad_text]
    # f.write(result_table)
    # f.write('\n'.join(result_list))
    f.write(accuracy_text)
    f.write(str(table))
    f.write(size_text)
    f.write(bad_text)
    f.close()

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
