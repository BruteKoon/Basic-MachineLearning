import scipy.io as spio
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

[fs, x] = wavfile.read('sound/JUNE_01_PHANTOMS/purdue_P2_unloaded_up_down.wav')

f, t, S = signal.stft(x, fs, window = 'hamm', nperseg=256, nfft = 1024)
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


