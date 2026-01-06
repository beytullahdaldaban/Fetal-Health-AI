import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def check_missing_values(df):
    """Eksik verileri raporlar."""
    return df.isnull().sum()

def handle_missing_values(df, strategy="mean"):
    """
    Eksik verileri seçilen stratejiye göre doldurur.
    strategy: 'mean' (ortalama), 'median' (medyan) veya 'drop' (silme)
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        return df.fillna(df.mean())
    elif strategy == "median":
        return df.fillna(df.median())
    return df

def clean_data(df):
    """Temel temizlik işlemleri (tekrar edenleri silme)."""
    # Tekrar eden satırları sil
    df = df.drop_duplicates()
    return df

def scale_features(X, method="Standard"):
    """
    Özellikleri ölçeklendirir (Normalization/Standardization).
    method: 'Standard' veya 'MinMax'
    """
    if method == "Standard":
        scaler = StandardScaler()
    elif method == "MinMax":
        scaler = MinMaxScaler()
    else:
        return X, None
    
    # Ölçeklenmiş veriyi DataFrame'e geri çevir (sütun isimlerini koru)
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler