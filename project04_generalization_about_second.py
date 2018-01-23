import scipy.io as spio
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import random
import tensorflow as tf

# "txt" -> array
text_array = np.loadtxt('txt/purdue_P2_unloaded_up_down.txt')
text_array = np.transpose(text_array)
fs = 44100

# specify length of time slice in sec
time_slice = 5
data_per_time_slice = int(fs*time_slice)

# for loop
for i in range(3):
	#select random number
	rand_num = random.randrange(0,text_array[1].shape[0] - data_per_time_slice)
	#make a random_array
	random_array = text_array[2][rand_num:rand_num+data_per_time_slice]
	
	#view graph
	f, t, S = signal.stft(random_array, fs, window = 'hamm', nperseg=256, nfft = 1024)
	S = np.abs(S)
	S = 20*np.log10(S + 1e-6)
	
	#normalization
	maximum = max(max(S.reshape([1, -1])))
	minimum = min(min(S.reshape([1, -1])))
	S_norm = (S - minimum)/(maximum-minimum)
	
	plt.pcolormesh(t, f, S_norm, cmap = cm.plasma)
	plt.title('STFT Magnitude')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
#	plt.show()
	
	
#	picture's array 
#	print (S_norm)
#	print(S_norm.shape) #(513,1724)

	#make data_set (1 array = 884412)
	if i is 0:
		data_set = S_norm.reshape(-1)
	else:
		data_set = np.vstack((data_set, S_norm.reshape(-1)))


print (data_set)
print (data_set.shape)

####################### CNN ##################################
########### INPUT_DATA = S_norm(2demension_array) #############################






