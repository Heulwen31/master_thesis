import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.utils.config import get_model_config

class ModelFactory:
    @staticmethod
    def get_model(model_type, **kwargs):
        """
        Returns an initialized model instance based on the model_type and parameters from config.
        kwargs can be used to override config parameters.
        """
        config = get_model_config()
        if model_type not in config:
            raise ValueError(f"Model type '{model_type}' not found in model.yml")
        
        params = config[model_type].copy()
        params.update(kwargs) # Override with any passed arguments
        
        if model_type == "xgboost":
            print(f"Initializing XGBoost with params: {params}")
            return xgb.XGBClassifier(**params)
        
        elif model_type == "lightgbm":
            print(f"Initializing LightGBM with params: {params}")
            return lgb.LGBMClassifier(**params)
        
        elif model_type == "catboost":
            print(f"Initializing CatBoost with params: {params}")
            return CatBoostClassifier(**params)
        
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported yet.")

def create_model(model_type, **kwargs):
    return ModelFactory.get_model(model_type, **kwargs)
