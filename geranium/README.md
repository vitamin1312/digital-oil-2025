Мощность активная, кВт: float64
Ток фазы А, A (ампер): float64
ЭЦН_rate_nom_sm3day: int64
ЭЦН_rate_opt_min_sm3day: int64
ЭЦН_rate_opt_max_sm3day: int64
НРХ_eff_area: float64
ЭЦН_rate_max_sm3day: float64
ПЭД_motor_nom_power: float64
НРХ_head_area: float64
НРХ_power_area: float64
ЭЦН_eff_max: float64
ПЭД_motor_amp_idle: float64
ПЭД_motor_nom_i: float64
Напряжение, АВ Вольт: float64
![Пример входных данных](image.png)
В последней ячейке блокнота заменяем пример на нужный
sample_input = {
    "Мощность активная, кВт": 180.5,
    "Ток фазы А, A (ампер)": 85.3,
    "ЭЦН_rate_nom_sm3day": 220,
    "ЭЦН_rate_opt_min_sm3day": 160,
    "ЭЦН_rate_opt_max_sm3day": 260,
    "НРХ_eff_area": 110.2,
    "ЭЦН_rate_max_sm3day": 300.0,
    "ПЭД_motor_nom_power": 160.0,
    "НРХ_head_area": 145.0,
    "НРХ_power_area": 175.5,
    "ЭЦН_eff_max": 0.62,
    "ПЭД_motor_amp_idle": 26.7,
    "ПЭД_motor_nom_i": 56.3,
    "Напряжение, АВ Вольт": 380.0
}