import ast
import pandas as pd
from typing import List


class ESP:
    """Класс для хранения данных об электроцентробежном насосе (ЭЦН)"""
    def __init__(self, data: dict):
        self.name = data.get('name')
        self.rate_nom_sm3day = data.get('rate_nom_sm3day')
        self.rate_opt_min_sm3day = data.get('rate_opt_min_sm3day')
        self.rate_opt_max_sm3day = data.get('rate_opt_max_sm3day')
        self.freq_Hz = data.get('freq_Hz')
        self.rate_points = data.get('rate_points', [])
        self.head_points = data.get('head_points', [])
        self.power_points = data.get('power_points', [])
        self.eff_points = data.get('eff_points', [])
        self.stages_max = data.get('stages_max')
        self.rate_max_sm3day = data.get('rate_max_sm3day')
        self.slip_nom_rpm = data.get('slip_nom_rpm')
        self.eff_max = data.get('eff_max')

    @property
    def get_rate_points(self) -> List[float]:
        return self.rate_points.copy()

    @property
    def get_head_points(self) -> List[float]:
        return self.head_points.copy()

    @property
    def get_power_points(self) -> List[float]:
        return self.power_points.copy()

    @property
    def get_eff_points(self) -> List[float]:
        return self.eff_points.copy()


class PED:
    """Класс для хранения данных о погружном электродвигателе (ПЭД)"""
    def __init__(self, data: dict):
        self.manufacturer = data.get('manufacturer')
        self.name = data.get('name')
        self.d_motor_mm = data.get('d_motor_mm')
        self.motor_nom_i = data.get('motor_nom_i')
        self.motor_amp_idle = data.get('motor_amp_idle')
        self.motor_nom_power = data.get('motor_nom_power')
        self.motor_nom_voltage = data.get('motor_nom_voltage')
        self.motor_nom_eff = data.get('motor_nom_eff')
        self.motor_nom_cosf = data.get('motor_nom_cosf')
        self.motor_nom_freq = data.get('motor_nom_freq')
        self.load_points = data.get('load_points', [])
        self.amperage_points = data.get('amperage_points', [])
        self.cosf_points = data.get('cosf_points', [])
        self.eff_points = data.get('eff_points', [])
        self.rpm_points = data.get('rpm_points', [])

    @property
    def get_load_points(self) -> List[float]:
        return self.load_points.copy()

    @property
    def get_amperage_points(self) -> List[float]:
        return self.amperage_points.copy()

    @property
    def get_cosf_points(self) -> List[float]:
        return self.cosf_points.copy()

    @property
    def get_eff_points(self) -> List[float]:
        return self.eff_points.copy()
    
    @property
    def get_rpm_points(self) -> List[float]:
        return self.rpm_points.copy()
    

class Inclinometry:
    """Класс для хранения данных инклинометрии"""
    def __init__(self, data: dict):
        self.measured = data.get('measured', [])
        self.absolute = data.get('absolute', [])

    @property
    def get_measured(self) -> List[float]:
        return self.measured.copy()

    @property
    def get_absolute(self) -> List[float]:
        return self.absolute.copy()


class Casing:
    """Класс для хранения данных об эксплуатационной колонне"""
    def __init__(self, data: dict):
        self.bottom_depth = data.get('bottom_depth')
        self.sections_info = data.get('sections_info', [])
        self.roughness = data.get('roughness')

    @property
    def get_sections_info(self) -> List[float]:
        return self.sections_info.copy()


class Tubing:
    """Класс для хранения данных о насосно-компрессорных трубах (НКТ)"""
    def __init__(self, data: dict):
        self.bottom_depth = data.get('bottom_depth')
        self.sections_info = data.get('sections_info', [])
        self.roughness = data.get('roughness')

    @property
    def get_sections_info(self) -> List[float]:
        return self.sections_info.copy()


class Separator:
    """Класс для хранения данных о сепараторе"""
    def __init__(self, data: dict):
        self.k_gas_sep = data.get('k_gas_sep')
        self.sep_name = data.get('sep_name')


def safe_parse_dict(s: str, field_name: str = "", well_id: int = -1) -> dict:
    if not isinstance(s, str):
        print(f"[Warning] Well {well_id}: поле '{field_name}' не является строкой (type={type(s)}).")
        return {}
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"[Warning] Well {well_id}: не удалось распарсить '{field_name}' — {e}")
        return {}

class Well:
    """Класс для хранения всех данных о скважине"""
    def __init__(self, well_id: int, data: dict):
        self.well_id = well_id

        self.esp = ESP(safe_parse_dict(data.get('Паспортные данные ЭЦН'), 'Паспортные данные ЭЦН', well_id))
        self.ped = PED(safe_parse_dict(data.get('Характеристики ПЭД'), 'Характеристики ПЭД', well_id))
        self._nclinometry = Inclinometry(safe_parse_dict(data.get('Инклинометрия'), 'Инклинометрия', well_id))
        self.casing = Casing(safe_parse_dict(data.get('ЭК'), 'ЭК', well_id))
        self.tubing = Tubing(safe_parse_dict(data.get('НКТ'), 'НКТ', well_id))

        self.packer = data.get('Пакер')
        self.stages = data.get('Количество ступеней насоса')

        sep_data = data.get('Сепаратор')
        self.separator = Separator(safe_parse_dict(sep_data, 'Сепаратор', well_id)) if pd.notna(sep_data) else None

        self.control_station_efficiency = data.get('КПД станции управления')
        self.transformer_efficiency = data.get('КПД трансформатора')
        self.pump_depth = data.get('Глубина установки насоса')
        self.cable_resistance = data.get('Удельное сопротивление кабеля')
        self.cable_length = data.get('Длина кабеля')
        self.gas_density = data.get('Относительная плотность газа')
        self.oil_density = data.get('Относительная плотность нефти')
        self.water_density = data.get('Относительная плотность воды')
        self.reservoir_pressure = data.get('Пластовое давление')
        self.reservoir_temperature = data.get('Пластовая температура')
        self.choke = data.get('Штуцер')
        self.linear_temperature = data.get('Линейная температура')
