import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

sample_data_path = mne.datasets.sample.data_path()  # Get the path to the sample dataset
raw_file = sample_data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'  # Construct the file path

# Load the data
raw = mne.io.read_raw_fif(str(raw_file), preload=True)  # Load the raw EEG data


print(raw.info['ch_names']) 


eog_channels = mne.pick_types(raw.info, meg=False, eeg=False, eog=True)
eog_channel_names = [raw.ch_names[i] for i in eog_channels]
print(f"EOG channels: {eog_channel_names}")


if eog_channel_names:
    eog_channel = eog_channel_names[0]  # Use the first EOG channel found
else:
    raise ValueError("No EOG channel found in the dataset")


raw.filter(1., 40., fir_design='firwin')  # Apply a bandpass filter from 1 to 40 Hz


picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')  # Select only EEG channels, excluding bad channels


ica = ICA(n_components=20, random_state=97, max_iter=800) 
ica.fit(raw, picks=picks)  


ica.plot_components()  # Plot the topographic maps of the ICA components


eog_inds, scores = ica.find_bads_eog(raw, ch_name=eog_channel)  # Find components related to eye blinks using the identified EOG channel

# Plot scores
ica.plot_scores(scores)  # Plot the correlation scores of the components with the EOG channel


ica.exclude = eog_inds  # Mark the identified components for exclusion

# Plot the sources
ica.plot_sources(raw)  # Plot the time series of the components

# Remove the components
reconstructed_raw = raw.copy()  # Create a copy of the original raw data
ica.apply(reconstructed_raw)  # Apply the ICA solution to the copy, excluding the identified components


raw.plot(title='Original data')  # Plot the original raw EEG data
reconstructed_raw.plot(title='Cleaned data')  # Plot the cleaned EEG data

plt.show()  # Display the plots
