from PIL import Image
from PIL import GifImagePlugin
import numpy
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.fftpack import fft
from numpy import fft
import wave
from scipy.io import wavfile # get the api
import struct


def show_info(aname, a):
    print "Array", aname
    print "shape:", a.shape
    print "dtype:", a.dtype
    print "min, max:", a.min(), a.max()
    print


def test():
    fname = "/home/wspek/Code/dev/eko2018/morse.wav"
    frate, data = wavfile.read(fname)
    data_size = len(data)

    wav_file = wave.open(fname, 'r')
    data = wav_file.readframes(data_size)
    wav_file.close()

    data = struct.unpack('{n}h'.format(n=data_size), data)
    data = np.array(data)

    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w))
    print(freqs.min(), freqs.max())
    # (-0.5, 0.499975)

    # Find the peak in the coefficients
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * frate)
    print(freq_in_hertz)
    # 439.8975

if __name__ == '__main__':
    for i in range(1, 6):
        image_object = Image.open('/home/wspek/Code/dev/eko2018/groups/{}/sum.png'.format(i))
        image_object = image_object.convert("L")  # Convert to greyscale
        histogram = image_object.histogram()
        test = 0

    test()

    rate, data = wavfile.read('/home/wspek/Code/dev/eko2018/morse.wav')
    show_info("data", data)

    fourier = fft.fft(data)

    # plt.plot(fourier, color='#ff7f00')
    # plt.xlabel('k')
    # plt.ylabel('Amplitude')
    # plt.show()

    n = len(data)
    fourier = fourier[0:(n / 2)]

    # scale by the number of points so that the magnitude does not depend on the length
    fourier = fourier / float(n)

    # calculate the frequency at each point in Hz
    freqArray = np.arange(0, (n / 2), 1.0) * (rate * 1.0 / n);

    plt.plot(freqArray / 1000, 10 * np.log10(fourier), color='#ff7f00', linewidth=0.02)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.show()


