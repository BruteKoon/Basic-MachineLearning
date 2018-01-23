import scipy.io as spio
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

text_array = np.loadtxt('txt/purdue_P2_unloaded_up_down.txt')
text_array = np.transpose(text_array)
fs = 44100

f, t, S = signal.stft(text_array[1], fs, window = 'hamm', nperseg=256, nfft = 1024)
S = np.abs(S)
S = 20*np.log10(S + 1e-6)

maximum = max(max(S.reshape([1, -1])))
minimum = min(min(S.reshape([1, -1])))
S_norm = (S - minimum)/(maximum-minimum)

plt.pcolormesh(t, f, S_norm, cmap = cm.plasma)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

print(S_norm.shape)




