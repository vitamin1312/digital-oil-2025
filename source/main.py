import os
import pandas as pd
from data_preparation import WellDataProcessor, check_filepath
from data_analysis import WellProductionModel
import config as cf

def main() ->  None:
    #  Парсинг и валидация данных
    print("Parsing and processing data...")
    data_processor = WellDataProcessor()

    processed_train_df = None
    processed_test_df = None

    if not os.path.exists(cf.CACHE_FOLDER):
        os.makedirs(cf.CACHE_FOLDER)

    if check_filepath(cf.TRAIN_DATA_CACHE_FILE, { ".csv" }):
        processed_train_df = pd.read_csv(cf.TRAIN_DATA_CACHE_FILE)
    else:
        processed_train_df = data_processor.process_data(cf.TRAIN_STATIC_DATA, cf.TRAIN_TM_DATA_FOLDER)
        processed_train_df.to_csv(cf.TRAIN_DATA_CACHE_FILE, index=False)

    if check_filepath(cf.TEST_DATA_CACHE_FILE, { ".csv" }):
        processed_test_df = pd.read_csv(cf.TEST_DATA_CACHE_FILE)
    else:
        processed_test_df = data_processor.process_data(cf.TEST_STATIC_DATA, cf.TEST_TM_DATA_FOLDER)
        processed_test_df.to_csv(cf.TEST_DATA_CACHE_FILE, index=False)

    # Обучение модели
    print("Training model...")
    if not os.path.exists(cf.MODELDIR):
        os.makedirs(cf.MODELDIR)

    model = WellProductionModel()
    model.train(processed_train_df)
    model.save_model(cf.MODEL_NAME)
    
    # Оценка модели
    print("Evaluating model...")
    mape, smape = model.evaluate(processed_test_df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        pass
