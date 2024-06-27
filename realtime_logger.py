import numpy as np
from sklearn.decomposition import FastICA
import time
from scipy.stats import zscore
from scipy.signal import butter, lfilter
from board import BoardManager
from brainflow.board_shim import BoardShim, BoardIds


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y

board_manager = BoardManager()
board_manager.setup_board()

sampling_rate = 125
chunk_size = 2 * sampling_rate

try:
    while True:
        time.sleep(2)

        data = board_manager.get_board_data()

        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)
        eeg_data = data[eeg_channels, -chunk_size:]

        eeg_data = bandpass_filter(eeg_data, 1.0, 40.0, sampling_rate, order=5)

        ica = FastICA(n_components=len(eeg_channels), random_state=97, max_iter=1000)
        sources = ica.fit_transform(eeg_data.T).T

        frontal_channels = [0, 1]
        frontal_channel_data = eeg_data[frontal_channels, :]
        eog_signal = zscore(frontal_channel_data.mean(axis=0))

        eog_correlation = np.array([np.corrcoef(eog_signal, source)[0, 1] for source in sources])

        threshold = 0.3
        eog_inds = np.where(np.abs(eog_correlation) > threshold)[0]

        if len(eog_inds) > 0:
            print(f"Blink detected at {time.strftime('%Y-%m-%d %H:%M:%S')} with components: {eog_inds}")

        mixing_matrix = ica.mixing_
        sources[eog_inds, :] = 0
        cleaned_eeg_data = np.dot(mixing_matrix, sources).T

except KeyboardInterrupt:
    print("Stopping the stream and releasing the session.")
    board_manager.stop_stream()
    board_manager.release_session()
