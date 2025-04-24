import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
from typing import Dict, Tuple, Optional


def train_debit_model(
        data: pd.DataFrame,
        target_col: str = "Дебит жидкости (объёмный), м3/сут",
        time_col: str = "tm_time",
        well_id_col: str = "well_id",
        water_cut_col = "Обводненность (объёмная), %",
        test_size: float = 0.2,
        model_params: Optional[dict] = None,
        save_model: bool = False,
        output_path: str = "."
) -> Dict:
    # 1. Валидация входных данных
    required = {target_col, time_col, well_id_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")

    # 2. Предобработка данных
    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df[(df[water_cut_col] > 0)]
    df = df.sort_values([well_id_col, time_col]).reset_index(drop=True)

    # 3. Подготовка признаков
    features = df.columns.difference([target_col, well_id_col, time_col])
    X, y = df[features], df[target_col]

    # 4. Временное разделение данных
    train_mask = np.zeros(len(df), dtype=bool)
    for _, group in df.groupby(well_id_col):
        split_idx = int(len(group) * (1 - test_size))
        train_mask[group.index[:split_idx]] = True

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    # 5. Конфигурация модели
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'early_stopping_rounds': 50,
        'random_state': 42,
        'eval_metric': ['mae', 'rmse']
    }
    params.update(model_params or {})

    # 6. Обучение модели
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=features.tolist())
    dtest = xgb.DMatrix(X_test, y_test, feature_names=features.tolist())

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=params['early_stopping_rounds'],
        verbose_eval=50
    )

    # 7. Расчет метрик
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    preds = model.predict(dtest)
    
    metrics = calculate_metrics(y_test.values, preds)

    # Добавляем недостающие ключи в метрики
    metrics.update({
        'target': target_col,
        'features': features.tolist()  # Сохраняем список фичей
    })

    # 8. Важность признаков
    importance = model.get_score(importance_type='gain')
    metrics['feature_importance'] = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 9. Сохранение модели
    if save_model:
        os.makedirs(output_path, exist_ok=True)
        model.save_model(os.path.join(output_path, 'debit_model.ubj'))
        meta = {
            'features': features.tolist(),
            'target': target_col,
            'metrics': metrics
        }
        with open(os.path.join(output_path, 'model_meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

    return {
        'model': model,
        'metrics': metrics,
        'data': (X_train, X_test, y_train, y_test)
    }


# def get_median(data: pd.DataFrame, features: list, well_id_col: str = "well_id") -> pd.DataFrame:
#     """
#     Вычисляет медианные значения только для указанных признаков.

#     Параметры:
#     data: Исходные данные
#     features: Список числовых колонок для расчета медианы
#     well_id_col: Колонка с идентификатором скважины

#     Возвращает:
#     DataFrame с медианными значениями
#     """
#     # Выбираем только нужные колонки
#     required_cols = [well_id_col] + features
#     return data[required_cols].groupby(well_id_col).median().reset_index()


# def predict_median_debits(model: xgb.Booster, meta: dict, new_data: pd.DataFrame) -> np.ndarray:
#     """Выполняет прогноз для медианных значений признаков."""
#     features = meta['features']
#     median_data = get_median(new_data, features)
#     dmatrix = xgb.DMatrix(median_data[features], feature_names=features)
#     return model.predict(dmatrix)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def custom_mape(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-6) -> float:
    """
    Вычисляет MAPE с защитой от деления на ноль через eps.
    
    Параметры:
    ----------
    y_true : array-like
        Фактические значения.
    y_pred : array-like
        Предсказанные значения.
    eps : float, optional
        Малое число для стабилизации знаменателя (по умолчанию 1e-6).
    
    Возвращает:
    -----------
    mape : float
        Средняя абсолютная процентная ошибка в процентах.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Проверка на одинаковую длину
    if len(y_true) != len(y_pred):
        raise ValueError("Длины y_true и y_pred должны совпадать")
    
    # Вычисление абсолютных ошибок с защитой от деления на ноль
    absolute_errors = np.abs(y_true - y_pred)
    denominators = np.maximum(np.abs(y_true), eps) 
    
    # Расчёт MAPE в процентах
    mape = 100 * np.mean(absolute_errors / denominators)
    
    return mape


if __name__ == "__main__":
    # Обучение модели
    df = pd.read_csv('data_frame.csv')
    results = train_debit_model(
        data=df,
        model_params={'n_estimators': 1500},
        save_model=True
    )


    # Получаем параметры из результатов
    target_col = results['metrics']['target']
    features = results['metrics']['features']

    # Загрузка тестовых данных
    test_df = pd.read_csv('test_df.csv')
    
    # Подготовка данных
    X_test = test_df[features]
    y_test = test_df[target_col].values

    # Прогнозирование
    model = results['model']

    dtest = xgb.DMatrix(X_test, feature_names=features)
    water_cut_col = "Обводненность (объёмная), %"

    test_preds = np.zeros(len(X_test))
    
    # Находим строки, где обводнённость > 0
    mask = X_test[water_cut_col] > 0
    # test_preds = []
    # Если есть строки для предсказания
    if mask.any():
        # Подготавливаем данные для модели
        predict_data = X_test.loc[mask, features]
        dmatrix = xgb.DMatrix(predict_data, feature_names=list(features))
        
        # Выполняем предсказание только для нужных строк
        test_preds[mask] = model.predict(dmatrix)
    # test_preds = results['model'].predict(dtest)

    # Расчет метрик
    mse = np.mean((y_test - test_preds) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_test - test_preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print("\nОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"MAPE: {custom_mape(y_test, test_preds):.2f}%")
    print(f"SMAPE: {smape(y_test, test_preds):.2f}%")
    # Прогноз для медианных значений
    # median_preds = predict_median_debits(results['model'], results['metrics'], test_df)
    # print("\nПРОГНОЗ ДЛЯ МЕДИАННЫХ ЗНАЧЕНИЙ:")
    # print(f"Средний дебит: {np.mean(median_preds):.2f} м³/сут")
    # print(pd.DataFrame(median_preds, columns=['Прогноз']).describe())