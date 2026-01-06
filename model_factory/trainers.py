from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def get_model(model_name, params=None):
    """
    İstenilen modeli ve hiperparametreleri döndürür.
    """
    if params is None:
        params = {}

    if model_name == "Random Forest":
        return RandomForestClassifier(**params, random_state=42)
    
    elif model_name == "XGBoost":
        # XGBoost için sınıf etiketlerini düzenlemek gerekebilir (0,1,2 gibi)
        return XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    elif model_name == "SVM":
        return SVC(**params, probability=True, random_state=42)
    
    elif model_name == "Logistic Regression":
        return LogisticRegression(**params, max_iter=1000, random_state=42)
    
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**params, random_state=42)
    
    else:
        raise ValueError(f"Model bulunamadı: {model_name}")