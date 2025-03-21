import librosa
import numpy as np

# Ses dosyalarını yükle
ses1, sr1 = librosa.load("ses1.wav", sr=None)
ses2, sr2 = librosa.load("ses2.wav", sr=None)

# Ses dosyalarının MFCC (Mel-Frequency Cepstral Coefficients) özelliklerini çıkar
mfcc1 = librosa.feature.mfcc(y=ses1, sr=sr1)
mfcc2 = librosa.feature.mfcc(y=ses2, sr=sr2)

# Özelliklerin benzerliğini ölç
benzerlik = np.corrcoef(mfcc1.mean(axis=1), mfcc2.mean(axis=1))[0, 1]

# Sonucu yazdır
print(f"Ses Benzerlik Skoru: {benzerlik}")

if benzerlik > 0.75:
    print("Sesler aynı kişiye ait olabilir.")
else:
    print("Sesler farklı kişilere ait olabilir.")
