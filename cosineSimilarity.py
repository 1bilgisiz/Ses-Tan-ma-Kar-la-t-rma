import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ses dosyasını yükleyin
ses1, sr1 = librosa.load("C:/Users/saniye.bilgic/Desktop/python/ses1.wav")
ses2, sr2 = librosa.load("C:/Users/saniye.bilgic/Desktop/python/ses2.wav")

# MFCC (Mel-Frequency Cepstral Coefficients) özelliklerini çıkarma
mfcc1 = librosa.feature.mfcc(y=ses1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=ses2, sr=sr2, n_mfcc=13)

# Özelliklerin ortalamasını alalım
mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)

# Kosinüs Benzerliği Hesaplama
similarity = cosine_similarity([mfcc1_mean], [mfcc2_mean])

# Sonucu yazdıralım
print("Cosine Similarity: ", similarity[0][0])

# Sonuç: 1'e yakınsa sesler benzer, 0'a yakınsa çok farklıdır.
