
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import zscore
from brainflow.board_shim import BoardShim, BoardIds
from board import BoardManager  

board_manager = BoardManager(dev=True)
board_manager.setup_board()


sampling_rate = 250  
chunk_size = 2 * sampling_rate  


try:
    while True:
        time.sleep(2) 

        
        data = board_manager.get_board_data()

        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)
        eeg_data = data[eeg_channels, -chunk_size:]  

    
        info = mne.create_info(ch_names=[f'EEG {i}' for i in range(len(eeg_channels))], sfreq=sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)

        raw.filter(1., 40., fir_design='firwin')

        ica = ICA(n_components=len(eeg_channels), random_state=97, max_iter=800)  # Use number of channels as n_components
        ica.fit(raw)

        frontal_channels = [0, 1]  
        frontal_channel_data = raw.copy().pick_channels([raw.ch_names[i] for i in frontal_channels]).get_data()
        eog_signal = zscore(frontal_channel_data.mean(axis=0))  # Combine and normalize the frontal channels

        sources = ica.get_sources(raw).get_data()
        eog_correlation = np.array([np.corrcoef(eog_signal, source)[0, 1] for source in sources])

        threshold = 0.3  
        eog_inds = np.where(np.abs(eog_correlation) > threshold)[0]

        # Log blinks to the console
        if len(eog_inds) > 0:
            print(f"\n\n\n\nBlink detected at {time.strftime('%Y-%m-%d %H:%M:%S')} with components: {eog_inds}\n\n\n\n")

        # Exclude the identified components
        ica.exclude = eog_inds

        # Remove the components
        reconstructed_raw = raw.copy()
        ica.apply(reconstructed_raw)

except KeyboardInterrupt:
    print("Stopping the stream and releasing the session.")
    board_manager.stop_stream()
    board_manager.release_session()
