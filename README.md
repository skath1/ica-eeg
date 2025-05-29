# 🧠 ICA-EEG

This project applies Independent Component Analysis (ICA) to electroencephalography (EEG) data to identify and remove artifacts, such as eye blinks and muscle movements. By decomposing EEG signals into statistically independent components, it facilitates the isolation of neural activity from noise, enhancing the quality of the data for further analysis.

The approach involves:
- **Decomposition**: Breaking down multichannel EEG data into independent components using ICA.
- **Artifact Identification**: Detecting components associated with common artifacts.
- **Signal Reconstruction**: Reconstructing the EEG signal after removing identified artifacts.

This process improves the signal-to-noise ratio, making the EEG data more suitable for subsequent analyses.
