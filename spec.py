import os
import random
import matplotlib.pyplot as plt

from constants import WORKING_DIR
from dataset.constants import POSITIVE_FOLDER_NAME, SEGMENTED_DATA_FOLDER, NEGATIVE_FOLDER_NAME, DATA_SUBFOLDER_LOCATION
from dataset.splitter import __apply_path
from interface.constants import WINDOW_SIZE
from models.generator import _load_data


def naive(data, file_name, path):
    # fig, ax = plt.subplots(1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax.axis('tight')
    # ax.axis('off')
    # ax.specgram(data.flatten(), NFFT=256, Fs=256)
    # plt.savefig(os.path.join(path, file_name + '.png'), bbox_inches='tight', transparent=True, pad_inches=0.0)
    import numpy as np
    from scipy import signal
    # plt.show()
    plt.figure(figsize=(3.00, 3.00), dpi=96)
    fig, axs = plt.subplots(18, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

    fmin = 0  # Hz
    fmax = 20  # Hz
    for i in range(0, 18):
        f, t, Sxx = signal.spectrogram(x=data[i].flatten(), fs=256, nperseg=256, )
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice, :][0]
        axs[i].pcolormesh(t, f, Sxx, )
        axs[i].axis('tight')
        axs[i].axis('off')


    # plt.show()
    # plt.figure(figsize=(224, 224))
    plt.savefig(os.path.join(path, file_name + '.png'), bbox_inches='tight', transparent=True, pad_inches=0.0, dpi=96)
    plt.close('all')


patients = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10", "chb15"]
# patients = ["chb10", "chb15"]
for patient in patients:
    print("Processing Patient {}".format(patient))
    positive_path = os.path.join(WORKING_DIR,
                                 DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                 POSITIVE_FOLDER_NAME)
    negative_path = os.path.join(WORKING_DIR,
                                 DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                 NEGATIVE_FOLDER_NAME)
    positive_files = list(filter(lambda x: "png" not in x, os.listdir(positive_path)))
    negative_files = list(filter(lambda x: "png" not in x, os.listdir(negative_path)))
    random.shuffle(negative_files)
    negative_files = negative_files[:len(positive_files)]

    for file in positive_files:
        file_path = __apply_path(file, positive_path, negative_path)
        data = _load_data(file_path, 18, (18, WINDOW_SIZE))
        naive(data, file, positive_path)

    for file in negative_files:
        file_path = __apply_path(file, positive_path, negative_path)
        data = _load_data(file_path, 18, (18, WINDOW_SIZE))
        naive(data, file, negative_path)
    print("Finish")
