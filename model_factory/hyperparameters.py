def get_hyperparameters(model_name):
    """
    Her model için arayüzde gösterilecek varsayılan parametreleri döndürür.
    """
    if model_name == "Random Forest":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "criterion": ["gini", "entropy"]
        }
    
    elif model_name == "XGBoost":
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    
    elif model_name == "SVM":
        return {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"]
        }
    
    elif model_name == "Logistic Regression":
        return {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        }

    elif model_name == "Decision Tree":
        return {
            "max_depth": [None, 10, 20, 30],
            "criterion": ["gini", "entropy"]
        }
        
    return {}