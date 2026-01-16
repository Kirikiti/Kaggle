import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor


class OOFGenerator:
    def __init__(self, *, train_df, target, test_df=None, n_splits=5, seed=42):
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.n_splits = n_splits
        self.seed = seed

        if self.target not in self.train_df.columns:
            raise ValueError(f"La columna objetivo '{self.target}' no está en train_df")

        self.models = {
            "catboost": [],
            "extratrees": [],
            "tabnet": []
        }

    def fit_predict(self):
        X = self.train_df.drop(columns=[self.target])
        y = self.train_df[self.target].values

        X_test = self.test_df.copy() if self.test_df is not None else None

        oof_preds = {k: np.zeros(len(self.train_df)) for k in self.models.keys()}
        test_preds = (
            {k: np.zeros(len(self.test_df)) for k in self.models.keys()}
            if X_test is not None else None
        )

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            print(f"\n===== FOLD {fold+1} =====")

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # CatBoost
            model_cb = CatBoostRegressor(
                loss_function="RMSE",
                depth=8,
                learning_rate=0.05,
                iterations=2000,
                verbose=False,
                random_seed=self.seed
            )
            model_cb.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
            self.models["catboost"].append(model_cb)

            oof_preds["catboost"][valid_idx] = model_cb.predict(X_valid)
            if X_test is not None:
                test_preds["catboost"] += model_cb.predict(X_test) / self.n_splits

            # ExtraTrees
            model_et = ExtraTreesRegressor(
                n_estimators=800,
                bootstrap=True,
                random_state=self.seed,
                n_jobs=-1
            )
            model_et.fit(X_train, y_train)
            self.models["extratrees"].append(model_et)

            oof_preds["extratrees"][valid_idx] = model_et.predict(X_valid)
            if X_test is not None:
                test_preds["extratrees"] += model_et.predict(X_test) / self.n_splits

            # TabNet
            model_tab = TabNetRegressor(
                seed=self.seed,
                verbose=0
            )
            model_tab.fit(
                X_train.values, y_train.reshape(-1, 1),
                eval_set=[(X_valid.values, y_valid.reshape(-1, 1))],
                patience=50,
                max_epochs=200
            )
            self.models["tabnet"].append(model_tab)

            oof_preds["tabnet"][valid_idx] = model_tab.predict(X_valid.values).reshape(-1)
            if X_test is not None:
                test_preds["tabnet"] += model_tab.predict(X_test.values).reshape(-1) / self.n_splits

        return oof_preds, test_preds, self.models
