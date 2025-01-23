import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# EEG dosyalarının isimleri
filename_resting = "gk1.vhdr"
filename_workload = "mw.vhdr"

# EEG verilerini yükleme
raw_resting = mne.io.read_raw_brainvision(filename_resting, preload=True)
raw_workload = mne.io.read_raw_brainvision(filename_workload, preload=True)

# Gereksiz GSR kanalını çıkarma
raw_resting.drop_channels(['gsr'])
raw_workload.drop_channels(['gsr'])

# Özellik çıkarma fonksiyonu
def extract_features(raw):
    data = raw.get_data()  # Veriyi numpy formatına dönüştür
    sfreq = int(raw.info['sfreq'])  # Örnekleme frekansı
    n_channels, n_samples = data.shape  # Kanal ve örnek boyutlarını al
    
    segment_length = sfreq  # 1 saniyelik segment
    n_segments = n_samples // segment_length  # Toplam segment sayısı
    segmented_data = np.zeros((n_segments, n_channels, 50))  # Her segment için 50 frekans
    
    for j in range(n_channels):  # Her kanal üzerinde iterasyon
        y = data[j, :]
        for i in range(n_segments):  # Her segment üzerinde iterasyon
            segment = y[i * segment_length:(i + 1) * segment_length]  # 1 saniyelik veri
            segment = segment - np.mean(segment)  # DC offset çıkarma
            fft_vals = np.fft.fft(segment) / segment_length  # Fourier dönüşümü
            power_spectrum = np.sqrt(np.abs(fft_vals) ** 2)  # Güç spektrumu
            segmented_data[i, j, :] = power_spectrum[:50]  # İlk 50 frekansı al
    
    # Her segment için özellik matrisine dönüştürme
    X = segmented_data.reshape(n_segments, n_channels, 50)  # (segment, kanal, frekans)
    return X

# Resting ve workload verileri için özellik çıkarma
X_resting = extract_features(raw_resting)
X_workload = extract_features(raw_workload)

# Etiketleri oluşturma
y_resting = np.zeros(X_resting.shape[0])  # Resting için etiket: 0
y_workload = np.ones(X_workload.shape[0])  # Workload için etiket: 1

# Verileri birleştirme
X = np.vstack([X_resting, X_workload])  # (örnek sayısı, kanal sayısı, frekans bileşenleri)
y = np.hstack([y_resting, y_workload])  # Etiketleri birleştir

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# LSTM modeli tanımlama
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),  # İlk LSTM katmanı
    Dropout(0.3),  # Aşırı öğrenmeyi önlemek için dropout
    Dense(32, activation='relu'),  # Gizli katman
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Çıkış katmanı (binary classification için sigmoid)
])

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Modeli eğitme
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,  # Eğitim döngü sayısı
    batch_size=32,  # Her adımda kullanılan örnek sayısı
    verbose=1
)

# Modelin performansını değerlendirme
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Doğruluğu: {accuracy:.2f}")

# Doğruluk ve kayıp eğrilerini çizme
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('lstm')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Karışıklık Matrisi
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Resting", "Workload"])

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
