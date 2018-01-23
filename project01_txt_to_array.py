import scipy.io as spio
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

text_array = np.loadtxt('txt/purdue_P2_unloaded_up_down.txt')
text_array = np.transpose(text_array)
print (text_array)



