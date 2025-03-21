# 🎤 Ses Karşılaştırma Projesi (`librosa` & `sklearn`)

Bu proje, **iki farklı ses kaydının aynı kişiye ait olup olmadığını analiz etmek** için `librosa` ve `sklearn` kütüphanelerini kullanarak **MFCC (Mel-Frequency Cepstral Coefficients)** özelliklerini çıkarır ve **Kosinüs Benzerliği** hesaplar.  

## 📌 Özellikler
✅ `librosa` ile ses dosyalarından MFCC özelliklerini çıkarma  
✅ `sklearn` ile Kosinüs Benzerliği hesaplama  
✅ **Seslerin benzerlik derecesini ölçerek** aynı kişiye ait olup olmadığını belirleme  

---

## 🛠 Kullanılan Teknolojiler
- **Python 3.x**
- **librosa** (Ses işleme ve analiz)
- **numpy** (Sayısal hesaplamalar)
- **scikit-learn** (Makine öğrenimi ve benzerlik hesaplama)

---

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Gerekli Kütüphaneleri Yükleme
İlk olarak, proje için gerekli Python kütüphanelerini yükleyin:  
```bash
pip install librosa numpy scikit-learn
