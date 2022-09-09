from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_numeric_categorical_preprocessor(
    numeric_features, categorical_features
):

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0,
        n_jobs=-1,
    )

    return preprocessor


class IdentityTransformer:
    def fit(self, x):
        # Fit the interface
        pass

    @staticmethod
    def transform(x):
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
