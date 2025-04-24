import numpy as np
import matplotlib.pyplot as plt


def time_data_plot(data: np.array, time: np.array, path_to_save: str):
    plt.plot(time, data, label='Исходные данные', alpha=0.7)
    plt.legend()
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.title('График')
    plt.grid(True, alpha=0.3)
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')


def plot_histogram(data, bins=10, title='Гистограмма', xlabel='Значения', ylabel='Частота'):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.savefig("gist", dpi=300, bbox_inches='tight')

def model(pred_file="y_pred.txt", val_file="y_val.txt"):
    """
    Строит график сравнения предсказанных и реальных значений
    
    Параметры:
    - pred_file: файл с предсказанными значениями (по умолчанию 'y_pred.txt')
    - val_file: файл с реальными значениями (по умолчанию 'y_val.txt')
    """
    # Чтение данных из файлов
    try:
        with open(pred_file) as f:
            pred = np.loadtxt(f)
        with open(val_file) as f:
            val = np.loadtxt(f)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return
    except Exception as e:
        print(f"Ошибка при чтении файлов: {e}")
        return

    # Проверка размеров данных
    if len(pred) != len(val):
        print(f"Предупреждение: Разные размеры данных (pred: {len(pred)}, val: {len(val)})")
        min_len = min(len(pred), len(val))
        pred = pred[:min_len]
        val = val[:min_len]
    # нормализация данных
    # val = normalize_csv.normalize_data(val)
    # pred = normalize_csv.normalize_data(pred)
    # Создание графика
    plt.figure(figsize=(12, 6))
    
    # График реальных значений
    plt.plot(val, label='Реальные значения (val)', color='blue', alpha=0.7, linewidth=2)
    
    # График предсказанных значений
    plt.plot(pred, label='Предсказанные значения (pred)', color='red', alpha=0.5, linewidth=1.5)
    
    # Настройка графика
    plt.title('Сравнение предсказанных и реальных значений')
    plt.xlabel('Номер наблюдения')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сохранение и отображение
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    # print(data_analysis.symmetric_mean_absolute_percentage_error(val, pred))

if __name__ == "__main__":
    model()
    