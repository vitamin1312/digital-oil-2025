import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100


smape_scorer = make_scorer(
    symmetric_mean_absolute_percentage_error,
    greater_is_better=False  # говорим, что меньшие значения лучше :contentReference[oaicite:0]{index=0}
)


class WellProductionModel:
    """Модель для прогнозирования дебита скважин"""

    def __init__(self):
        self.pipeline = None
        self.feature_columns = None

    def train(self, df: pd.DataFrame) -> None:
        """Обучение модели"""
        df = df.dropna() 
        X = df.drop(columns=["debit", "well_id"])
        y = df["debit"]

        self.feature_columns = X.columns.tolist()

        # Разделение на train и validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # Создание пайплайна
        self.pipeline = Pipeline([
            ("model", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, loss="huber"))
        ])

        self.pipeline.fit(X_train, y_train)

        # Оценка на валидации
        y_true = df["debit"].values
        y_pred = self.pipeline.predict(X_val)

        print("Мин. debit_true:", y_true.min())
        print("Кол-во нулевых или отрицательных debit_true:", (y_true <= 0).sum())

        # если есть нули, выведем несколько примеров
        zeros_idx = np.where(y_true <= 0)[0]
        print("Примеры y_pred при y_true=0:", y_pred[zeros_idx[:5]])

        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f"Validation MAPE: {mape:.4f}")
        smape = symmetric_mean_absolute_percentage_error(y_val, y_pred)
        print(f"Validation SMAPE: {smape:.4f}")

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оценка модели на новых данных"""
        df = df.dropna()
        X = df[self.feature_columns]
        y_true = df["debit"]
        y_pred = self.pipeline.predict(X)

        mape = mean_absolute_percentage_error(y_true, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

        result = df[["well_id"]].copy()
        result["debit_true"] = y_true
        result["debit_pred"] = y_pred
        result["error"] = np.abs((y_pred - y_true) / y_true)
        result["mape"] = mape
        result["smape"] = smape

        print(f"Evaluation MAPE: {mape:.4f}")
        print(f"Evaluation SMAPE: {smape:.4f}")

        return result

    def save_model(self, path: str) -> None:
        """Сохранение модели"""
        joblib.dump({
            "pipeline": self.pipeline,
            "features": self.feature_columns
        }, path)
        print(f'Model saved to {path}')

    def load_model(self, path: str) -> None:
        """Загрузка модели"""
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.feature_columns = data["features"]
        print(f'Model loaded from {path}')
