# EEG_MentalWorkloadDetection
Random forest and LSTM model for detecting mental workload stages using EEG Data

Dataset: EEG data from BrainVision files (.vhdr).

Channels: Data recorded from EEG channels, with unnecessary channels like GSR removed.

Segmentation: Division of EEG data into 1-second segments with a fixed sampling frequency (sfreq).

Preprocessing:
Removal of the mean in each segment to reduce noise.
Calculation of the power spectrum using Fourier Transform (FFT).

Frequency Analysis:
Utilization of the first 50 frequency components for each segment.
Combination of segments and channels into a feature matrix.
To visualize noise reduction, the mean of the power spectra of the segments was calculated across the channels.
The extracted features are modeled using a Random Forest Classifier (accuracy = 0.93) and an LSTM model (accuracy = 0.93).
Model evaluation is based on Accuracy, Precision, Recall, and F1 Score.

![Screenshot 2025-01-13 203210](https://github.com/user-attachments/assets/a867d43c-1d40-46f8-8b4e-69d0cdd3d592)
![Screenshot 2025-01-13 194810](https://github.com/user-attachments/assets/75a56690-b290-4224-b09b-7c70de9c61d5)
![Screenshot 2025-01-14 140248](https://github.com/user-attachments/assets/68f862e2-041a-42db-a5c4-ca9b547f6791)
