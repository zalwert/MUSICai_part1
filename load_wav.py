from scipy.io import wavfile
import os

samplerate, data = wavfile.read(os.getcwd() + r'\data\TEST\2.wav')

import numpy as np
X = np.array(data[100000:])
print(X)

if len(X.shape) > 1:
    X = [sum(i) for i in X]
X = np.trim_zeros(X, 'f')

data_points = 100000
song_time = 4*60+48

time_to_model = (data_points * song_time) / len(data)

print("Model is fed with: " + str(time_to_model) + " secs of each song")




