import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
import numpy as np
import time
from scipy.stats import zscore

class BoardManager:
    def __init__(self, dev=True):
        self.board = None
        self.dev = dev
        self.args = {
            "timeout": 0,
            "ip_port": 0,
            "ip_protocol": 0,
            "ip_address": "",
            "serial_port": "/dev/cu.usbserial-DP04VYKA",  # correct usb c port on collection laptop is /dev/cu.usbserial-DP04VYKA,
            "mac_address": "",
            "other_info": "",
            "serial_number": "",
            "board_id": BoardIds.CYTON_DAISY_BOARD,
            "file": "",
            "master_board": BoardIds.NO_BOARD
        }

    def setup_board(self):
        BoardShim.enable_dev_board_logger()
        params = self.create_params()
        if self.dev:
            self.args["board_id"] = BoardIds.SYNTHETIC_BOARD
        self.board = BoardShim(self.args["board_id"], params)
        self.board.prepare_session()
        self.board.start_stream()

    def create_params(self):
        params = BrainFlowInputParams()
        params.ip_port = self.args["ip_port"]
        params.serial_port = self.args["serial_port"]
        params.mac_address = self.args["mac_address"]
        params.other_info = self.args["other_info"]
        params.serial_number = self.args["serial_number"]
        params.ip_address = self.args["ip_address"]
        params.ip_protocol = self.args["ip_protocol"]
        params.timeout = self.args["timeout"]
        params.file = self.args["file"]
        params.master_board = self.args["master_board"]
        return params

    def stop_stream(self):
        if self.board:
            self.board.stop_stream()

    def release_session(self):
        if self.board:
            self.board.release_session()

    def get_board_data(self):
        if self.board:
            return self.board.get_board_data()
        return None

    def start_stream(self):
        if self.board:
            try:
                self.stop_stream()
                self.board.start_stream()
            except:
                self.board.start_stream()

# Initialize BoardManager and setup the board
board_manager = BoardManager()
board_manager.setup_board()

# Define parameters for live processing
sampling_rate = 250  # Sampling rate of the board, adjust accordingly
chunk_size = 5 * sampling_rate  # Process 5 seconds of data at a time

# Create an empty array to store processed data
processed_data = []

# Continuous processing loop
try:
    while True:
        time.sleep(5)  # Collect data for 5 seconds

        # Get the collected data
        data = board_manager.get_board_data()

        # Extract EEG channels (assuming 16 EEG channels for CYTON_DAISY_BOARD)
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)
        eeg_data = data[eeg_channels, -chunk_size:]  # Get the last chunk of data

        # Create MNE RawArray object
        info = mne.create_info(ch_names=[f'EEG {i}' for i in range(len(eeg_channels))], sfreq=sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)

        # Filter the data
        raw.filter(1., 40., fir_design='firwin')

        # Apply ICA
        ica = ICA(n_components=len(eeg_channels), random_state=97, max_iter=800)  # Use number of channels as n_components
        ica.fit(raw)

        # Manually create EOG-like signal using frontal channels
        frontal_channels = [0, 1]  # Example: using the first two channels as frontal channels
        frontal_channel_data = raw.copy().pick_channels([raw.ch_names[i] for i in frontal_channels]).get_data()
        eog_signal = zscore(frontal_channel_data.mean(axis=0))  # Combine and normalize the frontal channels

        # Correlate the EOG-like signal with the independent components
        sources = ica.get_sources(raw).get_data()
        eog_correlation = np.array([np.corrcoef(eog_signal, source)[0, 1] for source in sources])
        
        # Identify components to exclude based on correlation threshold
        threshold = 0.3  # Adjust this threshold as needed
        eog_inds = np.where(np.abs(eog_correlation) > threshold)[0]
        
        # Exclude the identified components
        ica.exclude = eog_inds

        # Plot correlation scores
        plt.figure()
        plt.bar(range(len(eog_correlation)), np.abs(eog_correlation))
        plt.axhline(y=threshold, color='r', linestyle='-')
        plt.title('Correlation of ICA components with EOG signal')
        plt.xlabel('ICA components')
        plt.ylabel('Correlation')
        plt.show()

        # Remove the components
        reconstructed_raw = raw.copy()
        ica.apply(reconstructed_raw)

        # Append the cleaned data to processed_data
        processed_data.append(reconstructed_raw.get_data())

        # Optionally, visualize the processed chunk
        reconstructed_raw.plot(title='Cleaned data chunk')
        plt.show()

except KeyboardInterrupt:
    print("Stopping the stream and releasing the session.")
    board_manager.stop_stream()
    board_manager.release_session()

# Combine all processed chunks into a single array
final_processed_data = np.concatenate(processed_data, axis=1)

# Create a final MNE RawArray object for the entire cleaned data
final_raw = mne.io.RawArray(final_processed_data, info)

# Plot the final cleaned data
final_raw.plot(title='Final Cleaned Data')
plt.show()
