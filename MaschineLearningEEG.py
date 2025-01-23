import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

filename_resting = "gk1.vhdr" #Diese können acuh mit gk1 und ga verwechselt verden
filename_workload = "mw.vhdr"

raw_resting = mne.io.read_raw_brainvision(filename_resting, preload=True) #preload=true schneller erreichbar durch speicher 
raw_workload = mne.io.read_raw_brainvision(filename_workload, preload=True)

#gsr löschen(unnötig)
raw_resting.drop_channels(['gsr'])
raw_workload.drop_channels(['gsr'])

def extract_features(raw):
    data = raw.get_data() #konvertiert zu numpy
    sfreq = int(raw.info['sfreq'])  #wie viele sample wird sammelt in einem sekunde
    n_channels, n_samples = data.shape #EEG kanäle und time bzw.sample Größe
    
    segment_length = sfreq #mit Größe einem Sekunde 
    n_segments = n_samples // segment_length
    segmentedData = np.zeros((n_segments, n_channels, 50))  #Matrix mit erste 50 Frekanz für jede Segment und Kanäle

    for j in range(n_channels):  #Iteration über Kanäle
        y = data[j, :]  #Alle Zeitreihen(zaman serisi) in einem Kanal
        for i in range(n_segments):  #Iteration über Segmente im Kanal
            segment = y[i * segment_length:(i + 1) * segment_length] #Data in einem Sekunde
            segment = segment - np.mean(segment)  #Wittelwert des Signals is gleich zu 0 für bessere Ergebnisse in Power Spectrum /Noise-Reduction
            fft_vals = np.fft.fft(segment) / segment_length #Fourier-Transform
            power_spectrum = np.sqrt(np.abs(fft_vals) ** 2) #Berechnung des Power Spectrums
            segmentedData[i, j, :] = power_spectrum[:50] #nur die erste 50 erhalten

    #Eigenschaften Matrix --> Segmente werden kombiniert für jedes Kanäle
    X = segmentedData.reshape(n_segments, -1)  #(segment, Kanal x Frekanz)
    return X

#feature exraction für jede Klasse
X_resting = extract_features(raw_resting)
X_workload = extract_features(raw_workload)

#Labels für Klassen weil sie nicht vorhanden sind
y_resting = np.zeros(X_resting.shape[0])  #Resting: 0
y_workload = np.ones(X_workload.shape[0])  #Workload: 1

#kombinieren von Eigenschaften und Label-Werte
X = np.vstack([X_resting, X_workload])
y = np.hstack([y_resting, y_workload])
from sklearn.preprocessing import StandardScaler

#Skalierung für effiziente Model-Training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Erstellung von Test- und Train-Klassen
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model-Training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

#Evaultaion des Models
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Karışıklık Matrisi Görselleştirme
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Resting", "Workload"])

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Özellik Önem Düzeylerini Görselleştirme
feature_importances = clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.title('Özellik Önem Düzeyleri')
plt.xlabel('Özellik İndeksi')
plt.ylabel('Önem Düzeyi')
plt.grid(True)
plt.show()

# Doğruluk ve Yanlış Sınıflandırmaların Görselleştirilmesi
labels = ["Resting", "Workload"]
accurate = (y_test == y_pred).sum()
misclassified = len(y_test) - accurate

plt.figure(figsize=(8, 6))
plt.bar(labels, [accurate, misclassified], color=['green', 'red'])
plt.title('Doğru ve Yanlış Sınıflandırmalar')
plt.ylabel('Örnek Sayısı')
plt.show()
