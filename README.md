# 🏥 Fetal Health AI: Yapay Zeka Destekli Klinik Karar Destek Sistemi

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Machine Learning](https://img.shields.io/badge/AI-XGBoost%20%7C%20RandomForest-green)

## 📖 Proje Hakkında
Bu proje, **Kardiyotokografi (CTG)** verilerini kullanarak anne karnındaki bebeğin sağlık durumunu analiz eden ve olası riskleri önceden tahmin eden bir **Makine Öğrenmesi (Machine Learning)** uygulamasıdır.

Sağlık profesyonellerine yardımcı olmak amacıyla geliştirilen bu sistem, **Açıklanabilir Yapay Zeka (XAI)** teknikleriyle modelin neden bu kararı verdiğini görselleştirir.

## 🚀 Özellikler
* **Canlı Tahmin Modülü:** Hasta verilerini girerek anlık risk analizi (Normal / Şüpheli / Patolojik).
* **Model Kıyaslama Arenası:** Random Forest, XGBoost, SVM gibi algoritmaları yarıştırıp en iyisini seçme imkanı.
* **İnteraktif Görselleştirme:** İbreli risk göstergeleri ve özellik önem (feature importance) grafikleri.
* **Dinamik Veri Analizi:** Veri setinin istatistiksel dağılımını inceleyen paneller.
* **Model Persistence:** Eğitilen en iyi modelin kaydedilmesi ve tekrar kullanılabilmesi.

## 🛠️ Kurulum ve Çalıştırma

Proje dosyalarını indirdikten sonra terminal üzerinden aşağıdaki adımları izleyin:

### 1. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
2. Uygulamayı Başlatın
Localhost'ta çalıştırmak için şu kodu yazıp Enter'a basmanız yeterlidir:

Bash

streamlit run main_app.py
Komutu çalıştırdıktan sonra tarayıcınızda otomatik olarak açılacaktır.

📂 Proje Mimarisi
Proje, sürdürülebilir Modüler Mimari prensibiyle geliştirilmiştir:

Plaintext

Fetal-Health-AI/
├── data_pipeline/          # Veri yükleme ve temizleme işlemleri
├── model_factory/          # Makine öğrenmesi modelleri (XGBoost, RF vb.)
├── evaluation/             # Performans ölçümü ve metrikler
├── visualization/          # Grafik çizim fonksiyonları
├── ui/                     # Streamlit arayüz kodları
└── main_app.py             # Ana uygulama dosyası
👨‍💻 Geliştirici
Beytullah Daldaban 🔗 GitHub Profilim

Bu proje, Görsel Programlama dersi final ödevi kapsamında geliştirilmiştir.