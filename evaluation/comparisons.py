import time
from evaluation.metrics import calculate_metrics

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Bir modeli eğitir, test eder ve performans sonuçlarını döndürür.
    """
    results = {}
    
    # 1. Modeli Eğit (Süre tutarak)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 2. Tahmin Yap
    y_pred = model.predict(X_test)
    
    # 3. Metrikleri Hesapla
    metrics = calculate_metrics(y_test, y_pred)
    
    # Sonuçları birleştir
    results.update(metrics)
    results["Training Time (sec)"] = training_time
    results["Model"] = model  # Eğitilmiş modeli sakla (Grafikler için lazım olacak)
    
    return results