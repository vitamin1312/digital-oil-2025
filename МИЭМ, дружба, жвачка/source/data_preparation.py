import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from well_data import *
import config as cf


def check_filepath(filepath: str, suffix_list: set) -> bool:
    if len(filepath) == 0:
        return False
    filepath = Path(filepath)
    if filepath.exists() and filepath.is_file() and filepath.suffix in suffix_list:
        with open(filepath, 'rb') as f:
            return f.read(1)
    return False


# Возвращает pd.DataFrame 
# с количеством строк равным количеству дней измерения Дебита
# Колонки - "tm_time" - среднее время замера за один день,
# колонки 1-12 - соответствующие данные, усредненные по дням до количества
def old_get_well_dynamic_data(filepath: str):
    bad_df = pd.read_csv(filepath)
    well_id = int(bad_df["well_id"][0])
    
    id_to_df = dict()
    for param_id in cf.ID_TO_PARAM_NAME:
        better_df = bad_df.loc[bad_df["param_id"] == param_id]
        better_df = better_df.sort_values("tm_time")
        better_df["tm_time"] = pd.to_datetime(better_df["tm_time"])
        better_df = better_df.drop(columns=["well_id", "param_id"])
        id_to_df[cf.ID_TO_PARAM_NAME[param_id]] = better_df
    
    first_day = (id_to_df[cf.ID_TO_PARAM_NAME[1]]["tm_time"].iloc[0])
    last_day = (id_to_df[cf.ID_TO_PARAM_NAME[1]]["tm_time"].iloc[-1])
    span = (last_day - first_day).days
    good_df = pd.DataFrame(index=range(span + 1), columns=list(id_to_df.keys()))

    datetimes = np.ndarray(span + 1, datetime)
    values = np.ndarray(span + 1, np.float64)
    for param_id in id_to_df:
        param_df = id_to_df[param_id]
        for day_num in range(span + 1):
            date = (first_day + timedelta(days=day_num))
            condition = (param_df['tm_time'].dt.month == date.month) & (param_df['tm_time'].dt.day == date.day)
            temp_df = param_df.loc[condition]
            if param_id == cf.ID_TO_PARAM_NAME[1]:
                datetimes[day_num] = temp_df["tm_time"].mean()
            values[day_num] = temp_df["tm_value"].mean()
        good_df[param_id] = values

    return well_id, good_df

def get_well_dynamic_data(filepath: str):
    # Читаем нужные данные
    df = pd.read_csv(
        filepath,
        usecols=["well_id", "param_id", "tm_time", "tm_value"],
        parse_dates=["tm_time"],
        dtype={"param_id": "int8", "well_id": "int32"}
    )

    # Получаем well_id из первой строки
    well_id = int(df["well_id"].iloc[0])

    # Добавляем колонку даты без времени для группировки
    df["date"] = df["tm_time"].dt.normalize()

    # Группируем по дате и param_id
    grouped = (
        df.groupby(["date", "param_id"], observed=True)
          .agg({"tm_value": "mean"})
          .reset_index()
    )

    # Пивотируем: строки — даты, колонки — param_id
    pivot_df = grouped.pivot(index="date", columns="param_id", values="tm_value")

    # Переименовываем столбцы: param_id → имена параметров
    pivot_df.columns = [cf.ID_TO_PARAM_NAME.get(pid, f"param_{pid}") for pid in pivot_df.columns]

    # Сброс индекса, чтобы даты не были индексом, а просто столбцом (и удалим этот столбец)
    pivot_df = pivot_df.reset_index(drop=True)

    return well_id, pivot_df


def normalize_data(data: np.array, window_size=5, threshold=3) -> np.array:
    """Нормализация данных"""
    
    cleaned_data = np.array(data, dtype=float).copy()  
    for i in range(len(data)):
        window = data[max(0, i-window_size):i+window_size]
        
        if len(window) == 0:
            continue
        
        #считаем медиану
        median = np.nanmedian(window)
        if np.isnan(median):
            mean = np.nanmedian(cleaned_data[max(0, i-window_size):i+window_size])


        mad = 1.4826 * np.median(np.abs(window - median))
        
        if np.isnan(data[i]) or (mad > 0 and np.abs(data[i] - median) > threshold * mad):
            cleaned_data[i] = median
        if np.isnan(cleaned_data[i]):
            cleaned_data[i] = median
        if np.isnan(cleaned_data[i]):
            cleaned_data[i] = cleaned_data[i-1]
        if np.isnan(cleaned_data[i]):
            cleaned_data[i] = np.mean(data)
    
    return cleaned_data


class DataPreprocessor:
    """Класс для предобработки данных телеметрии и статических данных скважин"""
    
    def __init__(self):
        pass

    def _extract_static_features(self, well: 'Well') -> Dict[str, float]:
        """Извлекает статические признаки из объекта Well"""
        features = {
            'well_id': well.well_id,
            'esp_rate_nom': well.esp.rate_nom_sm3day,
            'esp_rate_opt_min': well.esp.rate_opt_min_sm3day,
            'esp_rate_opt_max': well.esp.rate_opt_max_sm3day,
            'esp_freq': well.esp.freq_Hz,
            'esp_eff_max': well.esp.eff_max,
            'ped_nom_power': well.ped.motor_nom_power,
            'ped_nom_eff': well.ped.motor_nom_eff,
            'ped_nom_voltage': well.ped.motor_nom_voltage,
            'ped_nom_freq': well.ped.motor_nom_freq,
            'ped_nom_i': well.ped.motor_nom_i,
            'n_stages': well.stages,
            'pump_depth': well.pump_depth,
            'cable_length': well.cable_length,
            'cable_resistance': well.cable_resistance,
            'control_station_eff': well.control_station_efficiency,
            'transformer_eff': well.transformer_efficiency,
            'gas_density': well.gas_density,
            'oil_density': well.oil_density,
            'water_density': well.water_density,
            'reservoir_pressure': well.reservoir_pressure,
            'reservoir_temperature': well.reservoir_temperature,
            'linear_temperature': well.linear_temperature,
            'has_packer': int(well.packer == 'ИСТИНА' or well.packer is True),
            'has_separator': int(well.separator is not None),
        }

        # Усреднение параметров инклинометрии, если они есть
        if well._nclinometry.measured:
            features['incl_avg'] = np.mean(well._nclinometry.measured)
        else:
            features['incl_avg'] = np.nan

        return features

    def fit_transform(self, wells: Dict[int, 'Well'], telemetry_df: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Объединяет телеметрию и статические данные, возвращает итоговый датафрейм
        """
        """
        Объединяет телеметрию и статические данные по каждой скважине
        """
        combined_rows = []

        for well_id, well in wells.items():
            telemetry = telemetry_df.get(well_id)
            if telemetry is None or telemetry.empty:
                print(f"[Warning] Нет телеметрии для скважины {well_id}, пропускаем.")
                continue

            static_features = self._extract_static_features(well)

            # Дублируем статические признаки для каждой строки телеметрии
            static_df = pd.DataFrame([static_features] * len(telemetry))
            telemetry = telemetry.reset_index(drop=True)
            static_df = static_df.reset_index(drop=True)

            full_df = pd.concat([telemetry, static_df], axis=1)
            full_df['well_id'] = well_id
            combined_rows.append(full_df)

        if not combined_rows:
            print("[Error] Не найдено ни одной валидной скважины с телеметрией.")
            return pd.DataFrame()

        result_df = pd.concat(combined_rows, ignore_index=True)

        # Заполняем пропуски (при необходимости можно заменить на другую стратегию)
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')

        return result_df


class WellDataParser:
    """Класс для парсинга данных о скважинах из Excel файла"""
    def __init__(self):
        self.wells: dict = {}

    def parse(self, filepath: str) -> dict:
        if not check_filepath(filepath, cf.ALLOWED_STATIC_DATA_FORMATS):
            return
        
        self.wells.clear()

        df = pd.read_excel(filepath)
        for _, row in df.iterrows():
            well_id = row['ID скважины']
            well_data = row.to_dict()
            self.wells[well_id] = Well(well_id, well_data)
        
        return self.wells

    def get(self, well_id: int) -> Well:
        if well_id in self.wells:
            return self.wells[well_id]
        return None


class TelemetryDataParser:
    def __init__(self):
        self.telemetries: dict = {}
        self.wells_files = {}

    def __parse_one(self, filepath: str) -> int | pd.DataFrame:
        if not check_filepath(filepath, cf.ALLOWED_TM_DATA_FORMATS):
            return -1, None
        
        well_id, data = get_well_dynamic_data(filepath)
        data.select_dtypes(include=["number"])
        for i in range(1, len(data)):
            data.iloc[i, :] = normalize_data(data.iloc[i])

        return well_id, data

    def parse(self, folderpath: str) -> dict:
        if not os.path.isdir(folderpath):
            print(f"Папка {folderpath} не существует!")
            return
        
        self.telemetries.clear()

        # Проходим по всем файлам в папке
        for filename in tqdm(os.listdir(folderpath)):
            filepath = os.path.join(folderpath, filename)
            
            well_id, df = self.__parse_one(filepath)

            if well_id != -1 and df is not None:
                self.telemetries[well_id] = df

        return self.telemetries
    
    def parse(self, folderpath: str) -> dict:
        if not os.path.isdir(folderpath):
            print(f"Папка {folderpath} не существует!")
            return
        
        self.telemetries.clear()
        self.wells_files.clear()

        # Проходим по всем файлам в папке
        for filename in tqdm(os.listdir(folderpath)):
            filepath = os.path.join(folderpath, filename)
            
            well_id, df = self.__parse_one(filepath)
            newpath = cf.CACHE_FOLDER + cf.slash + "tmp" + cf.slash +os.path.basename(filepath)
            df.to_csv(newpath, index=False)

            self.wells_files[well_id] = newpath

        for well_id in tqdm(self.wells_files):
            self.telemetries[well_id] = pd.read_csv(self.wells_files[well_id])

        return self.telemetries

    def get(self, well_id: int) -> pd.DataFrame:
        if well_id in self.telemetries:
            return self.telemetries[well_id]
        return None
    

class WellDataProcessor:
    """Класс для обработки данных скважин, объединяющий парсинг и предобработку"""
    
    def __init__(self):
        self.static_parser = WellDataParser()
        self.telemtry_parser = TelemetryDataParser()

        self.preprocessor = DataPreprocessor()
        
    def process_data(self, static_path: str, telemetry_path: str) -> pd.DataFrame:
        wells = self.static_parser.parse(static_path)
        telemetry = self.telemtry_parser.parse(telemetry_path)

        return self.preprocessor.fit_transform(wells, telemetry)
