class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    A leakage-safe target encoder using KFold averaging.
    Works with scikit-learn Pipelines.
    """

    def __init__(self, cols=None, n_splits=5, smoothing=1.0, random_state=42):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_ = None
        self.mapping_ = {}

    def fit(self, X, y):
        X = X.copy()
        y = y.values if isinstance(y, pd.Series) else y

        if self.cols is None:
            self.cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        self.global_mean_ = np.mean(y)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        self.mapping_ = {col: {} for col in self.cols}

        for col in self.cols:
            col_values = X[col].values

            for train_idx, val_idx in kf.split(X):
                train_y = y[train_idx]
                train_col = col_values[train_idx]

                means = pd.DataFrame({
                    "col": train_col,
                    "y": train_y
                }).groupby("col")["y"].mean()

                for category, mean_val in means.items():
                    if category not in self.mapping_[col]:
                        self.mapping_[col][category] = []
                    self.mapping_[col][category].append(mean_val)

            # average folds + smoothing
            for category, values in self.mapping_[col].items():
                avg = np.mean(values)
                smoothed = (avg * len(values) + self.global_mean_ * self.smoothing) / (len(values) + self.smoothing)
                self.mapping_[col][category] = smoothed

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.cols:
            X[col] = X[col].map(self.mapping_[col]).fillna(self.global_mean_)

        return X
