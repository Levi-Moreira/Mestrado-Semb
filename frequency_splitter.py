from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# frequency_bands = [(4, 7), (7, 12), (12, 19), (19, 30), (30, 40)]
#
# array = np.load("/Users/levialbuquerque/PycharmProjects/semb/data/chb15/negative/negative_4697.npy")
#
# splits = np.zeros((len(frequency_bands), array.shape[1]), dtype=np.float)
#
# for index, band in enumerate(frequency_bands):
#     low, high = band
#     splits[index] = butter_bandpass_filter(array[0], low, high, 256, order=3)
#
# T = 5
# nsamples = T * 256
# t = np.linspace(0, T, nsamples, endpoint=False)
# plt.plot(t, splits[0], label='Filtered signal (%s-%s Hz)' % frequency_bands[0])
# plt.plot(t, splits[1], label='Filtered signal (%s-%s Hz)' % frequency_bands[1])
# plt.plot(t, splits[2], label='Filtered signal (%s-%s Hz)' % frequency_bands[2])
# plt.plot(t, splits[3], label='Filtered signal (%s-%s Hz)' % frequency_bands[3])
# plt.plot(t, splits[4], label='Filtered signal (%s-%s Hz)' % frequency_bands[4])
# plt.plot(t, array[0], label='Original')
# plt.legend(loc='upper left')
# plt.show()
