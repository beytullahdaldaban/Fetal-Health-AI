# 🏥 Fetal Health AI: Yapay Zeka Destekli Klinik Karar Destek Sistemi

![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit&logoColor=white) ![Machine Learning](https://img.shields.io/badge/AI-XGBoost%20%7C%20RandomForest-green)

## 📖 Proje Hakkında

Bu proje, **Kardiyotokografi (CTG)** verilerini kullanarak anne karnındaki bebeğin sağlık durumunu analiz eden ve olası riskleri önceden tahmin eden bir **Makine Öğrenmesi** uygulamasıdır.

Sağlık profesyonellerine yardımcı olmak amacıyla geliştirilen bu sistem, **Açıklanabilir Yapay Zeka (XAI)** teknikleriyle modelin neden bu kararı verdiğini görselleştirir.

## 🚀 Özellikler

- **Canlı Tahmin Modülü:** Hasta verilerini girerek anlık risk analizi (Normal / Şüpheli / Patolojik)
- **Model Kıyaslama Arenası:** Random Forest, XGBoost, SVM gibi algoritmaları yarıştırıp en iyisini seçme
- **İnteraktif Görselleştirme:** Risk göstergeleri ve özellik önem (feature importance) grafikleri
- **Dinamik Veri Analizi:** Veri setinin istatistiksel dağılımını inceleyen paneller
- **Model Persistence:** Eğitilen en iyi modelin kaydedilip tekrar kullanılabilmesi

## 🛠️ Kurulum ve Çalıştırma

```
pip install -r requirements.txt
streamlit run main_app.py
```

Komut çalıştıktan sonra tarayıcınızda otomatik olarak açılır.

## 📂 Proje Mimarisi

```
Fetal-Health-AI/
├── data_pipeline/          # Veri yükleme ve temizleme işlemleri
├── model_factory/          # Makine öğrenmesi modelleri (XGBoost, RF vb.)
├── evaluation/              # Performans ölçümü ve metrikler
├── visualization/           # Grafik çizim fonksiyonları
├── ui/                      # Streamlit arayüz kodları
└── main_app.py              # Ana uygulama dosyası
```

## 👨‍💻 Geliştirici

**Beytullah Daldaban** — [GitHub Profilim](https://github.com/beytullahdaldaban)

Bu proje, Görsel Programlama dersi final ödevi kapsamında geliştirilmiştir.
