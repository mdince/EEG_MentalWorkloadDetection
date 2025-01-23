import numpy as np
import mne
import matplotlib.pyplot as plt

filename = "mw.vhdr" #diese können mit gk1 und ga verwechselt werden
raw = mne.io.read_raw_brainvision(filename, preload=True) #preload=true schneller erreichbar durch speicher 


raw.drop_channels(['gsr']) #gsr löschen(unnötig)

data = raw.get_data() #konvertiert zu numpy 
sfreq = int(raw.info['sfreq']) #wie viele sample wird sammelt in einem sekunde
n_channels, n_samples = data.shape #EEG kanäle und time bzw. sample Größe

segment_length = sfreq #mit Größe einem Sekunde 
n_segments = n_samples // segment_length
segmentedData = np.zeros((n_segments, n_channels, 50)) #Matrix mit erste 50 Frekanz für jede Segment und Kanäle

for j in range(n_channels): #Iteration über Kanäle
    y = data[j, :] #Alle Zeitreihen(zaman serisi) in einem Kanal
    for i in range(n_segments): #Iteration über Segmente im Kanal
        segment = y[i * segment_length:(i + 1) * segment_length] #Data in einem Sekunde
        segment = segment - np.mean(segment) #Wittelwert des Signals is gleich zu 0 für bessere Ergebnisse in Power Spectrum /Noise-Reduction
        fft_vals = np.fft.fft(segment) / segment_length #Fourier-Transform
        power_spectrum = np.sqrt(np.abs(fft_vals) ** 2) #Berechnung des Power Spectrums
        segmentedData[i, j, :] = power_spectrum[:50] #nur die erste 50 erhalten

lastData = np.mean(segmentedData, axis=0) #Mittelwert des Power-Spektrums/ Noise-Reduction

#Visualisierung
y_min = np.min(lastData)
y_max = np.max(lastData)

freqs = np.fft.fftfreq(segment_length, d=1/sfreq)[:50]
fig, axes = plt.subplots(4, 4, figsize=(15, 10))

for j, ax in enumerate(axes.flat):
    if j < n_channels:
        ax.plot(freqs, lastData[j, :], label=f'{raw.ch_names[j]}', color='b')
        ax.set_title(f'{raw.ch_names[j]}')
        ax.set_ylabel("Power Density")
        ax.set_xlabel("Frequency (hz)")
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_ylim(y_min, y_max) 
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()


