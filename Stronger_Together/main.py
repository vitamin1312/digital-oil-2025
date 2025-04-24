import pandas as pd
import numpy as np
import os
import xgboost as xgb
from typing import Tuple

# Словарь для преобразования ID параметров в названия
id_to_param_name = {
    1: "Дебит жидкости (объёмный), м3/сут",
    2: "Обводненность (объёмная), %",
    3: "Рбуф, атм",
    4: "Давление линейное, атм",
    5: "Давление на приеме насоса, атм",
    6: "Загрузка ПЭД, %",
    7: "Попутный газ",
    8: "Частота вращения, Герц",
    9: "Ток фазы А, A (ампер)",
    10: "Мощность активная, кВт",
    11: "Напряжение, АВ Вольт",
    12: "P затрубное, атм",
}


def merge_csv(path: str) -> pd.DataFrame:
    """Объединяет все CSV-файлы из указанной директории в один DataFrame."""
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def new_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает данные:
    1. Преобразует время к ежедневному формату
    2. Создает сводную таблицу с параметрами по колонкам
    3. Добавляет недостающие колонки
    4. Заполняет пропуски
    """
    df = df.copy()
    # Нормализация времени до ежедневного формата
    df['tm_time'] = pd.to_datetime(df['tm_time']).dt.normalize()

    # Создание сводной таблицы
    pivot_df = (df
                .groupby(['well_id', 'tm_time', 'param_id'])['tm_value']
                .mean()
                .unstack()
                .rename(columns=id_to_param_name)
                )

    # Добавление недостающих колонок и сортировка
    full_columns = ['well_id', 'tm_time'] + list(id_to_param_name.values())
    return (pivot_df
    .reindex(columns=id_to_param_name.values())  # Добавляем отсутствующие колонки
    .ffill()  # Заполняем пропуски предыдущими значениями
    .reset_index()
    .sort_values(['well_id', 'tm_time'])
    [full_columns]  # Переупорядочиваем колонки
    )


def remove_outliers(df: pd.DataFrame, threshold: float = 1.5,
                    ignore: list = None) -> pd.DataFrame:
    """
    Удаляет выбросы методом IQR.
    Игнорирует указанные колонки (по умолчанию: нечисловые и временные).
    """
    ignore = ignore or ['well_id', 'tm_time']
    numeric = df.select_dtypes(np.number).columns.difference(ignore)

    # Вычисление границ выбросов
    q1 = df[numeric].quantile(0.1)
    q3 = df[numeric].quantile(0.9)
    iqr = q3 - q1

    # Фильтрация выбросов
    filtered = df[~((df[numeric] < (q1 - threshold * iqr)) |
                    (df[numeric] > (q3 + threshold * iqr))).any(axis=1)]

    return filtered.dropna().reset_index(drop=True)


def normalize_df(df: pd.DataFrame, method: str = 'minmax',
              exclude: list = None) -> pd.DataFrame:
    """
    Нормализует данные:
    - 'minmax': приводит значения к диапазону [0, 1]
    - 'zscore': стандартизирует данные (среднее=0, std=1)
    """
    exclude = exclude or ['well_id', 'tm_time']
    numeric = df.select_dtypes(np.number).columns.difference(exclude)

    if method == 'minmax':
        df[numeric] = (df[numeric] - df[numeric].min()) / (df[numeric].max() - df[numeric].min())
    elif method == 'zscore':
        df[numeric] = (df[numeric] - df[numeric].mean()) / df[numeric].std()
    return df


def save_df(df: pd.DataFrame, path: str, name: str) -> None:
    """Сохраняет DataFrame в CSV файл."""
    df.to_csv(f"{os.path.join(path, name)}.csv", index=False)

# def get_median(df: pd.DataFrame) -> float: return df['Дебит жидкости (объёмный), м3/сут'].median()

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
    
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# def predict_debit(new_data:pd.DataFrame) -> np.ndarray:
#     model = xgb.Booster()
#     model.load_model('debit_model.ubj')
#     # new_data = pd.read_csv(path)
#     target_col = "Дебит жидкости (объёмный), м3/сут"
#     time_col = "tm_time"
#     well_id_col = "well_id"
#     features = list(new_data.columns.difference([target_col, well_id_col, time_col]))

#    # Подготовка данных
#     X_test = new_data[features]
#     #y_test = new_data[target_col].values

#     # Прогнозирование


#     water_cut_col = "Обводненность (объёмная), %"

#     test_preds = np.zeros(len(X_test))

#     # Находим строки, где обводнённость > 0
#     mask = X_test[water_cut_col] > 0

#     # Если есть строки для предсказания
#     if mask.any():
#         # Подготавливаем данные для модели
#         predict_data = X_test.loc[mask, features]
#         dmatrix = xgb.DMatrix(predict_data, feature_names=list(features))
        
#         # Выполняем предсказание только для нужных строк
#         test_preds[mask] = model.predict(dmatrix)

#     return test_preds

def predict_debit(new_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Прогнозирует дебит жидкости с учётом обводнённости.
    
    Параметры:
    ----------
    new_data : pd.DataFrame
        Входные данные для прогнозирования
        
    Возвращает:
    -----------
    Tuple[np.ndarray, np.ndarray]
        Кортеж из двух массивов: (well_ids, predictions)
        где:
        - well_ids - идентификаторы скважин
        - predictions - предсказанные значения дебита
    """
    # Загрузка модели
    model = xgb.Booster()
    model.load_model('debit_model.ubj')
    
    # Определение колонок
    target_col = "Дебит жидкости (объёмный), м3/сут"
    time_col = "tm_time"
    well_id_col = "well_id"
    water_cut_col = "Обводненность (объёмная), %"
    
    # Проверка наличия необходимых колонок
    required_cols = {target_col, time_col, well_id_col, water_cut_col}
    missing_cols = required_cols - set(new_data.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
    
    # Получение признаков и well_id
    features = list(new_data.columns.difference([target_col, well_id_col, time_col]))
    well_ids = new_data[well_id_col].values
    
    # Инициализация массива предсказаний
    predictions = np.zeros(len(new_data))
    
    # Находим строки с ненулевой обводнённостью
    mask = new_data[water_cut_col] > 0
    
    # Прогнозирование только для строк с ненулевой обводнённостью
    if mask.any():
        predict_data = new_data.loc[mask, features]
        dmatrix = xgb.DMatrix(predict_data, feature_names=features)
        predictions[mask] = model.predict(dmatrix)
    
    return well_ids, predictions

def metrics(y_test, test_preds):
    mse = np.mean((y_test - test_preds) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_test - test_preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    c_mape_res = custom_mape(y_test, test_preds)
    smape_res = smape(y_test, test_preds)

    return rmse, r2, c_mape_res, smape_res

if __name__ == "__main__":
    folder_path = 'C:/Users/79277/Desktop/ucheba/HackatON/test_2'
    data = merge_csv(folder_path)
    data = new_table(data)
    data.drop(columns=["Дебит жидкости (объёмный), м3/сут"])

    result = predict_debit(data)
    # print(result)
    rmse, r2, c_mape_res, smape_res = metrics(data["Дебит жидкости (объёмный), м3/сут"], result[1])
    print(rmse, r2, c_mape_res, smape_res)
    #print(list(zip(result[0],result[1])))



# test_df = merge_csv('C:/Users/itsvo/Downloads/Telegram Desktop/1 день/1 день/test_dataset_1/tm_data')
# test_df = new_table(test_df)
# print(test_df)
# test_df = remove_outliers(test_df)
# print(len(test_df))
# test_df = normalize_df(test_df)
# print(len(test_df))
# save_df(test_df, 'C:/Users/itsvo/Downloads/Telegram Desktop/1 день/1 день', 'test_df')
# print(len(test_df))
# print(get_median(test_df))