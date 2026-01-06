import pandas as pd
import os

def load_data(file_path):
    """
    CSV dosyasını yükler ve DataFrame olarak döndürür.
    """
    # Dosya var mı kontrol et
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HATA: Veri dosyası bulunamadı! Yol: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Veri başarıyla yüklendi. Toplam satır: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Veri yüklenirken hata oluştu: {e}")
        return None