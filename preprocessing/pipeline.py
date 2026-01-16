from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .kfold_target_encoder import KFoldTargetEncoder

def create_preprocessing_pipeline(n_splits=5, seed=42):
    return Pipeline([
        ("kfold_te", KFoldTargetEncoder(n_splits=n_splits, seed=seed)),
        ("scaler", StandardScaler())
    ])
