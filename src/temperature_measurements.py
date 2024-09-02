import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.regularizers import L1L2, L1, L2
from keras.layers import Dense
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from tensorflow.keras.saving import register_keras_serializable
import joblib
import math
import os
import keras

# Clear any existing custom objects
keras.saving.get_custom_objects().clear()

# def custom_accuracy_item(y_real, y_pred):
#     # A fórmula é |1 - (y_pred / y_true)| < 0.1, retornando um tensor booleano
#     # (y_pred * 1) / y_real = x
#     condition = tf.math.abs(1 - (y_pred / y_real)) < 0.1
#     # Convertendo o tensor booleano em float (True se torna 1.0, False se torna 0.0)
#     as_float = tf.cast(condition, tf.float32)
#     # Calculando a média para obter a acurácia
#     return as_float

# @register_keras_serializable(package="Custom", name="custom_accuracy")
# def custom_accuracy(y_real, y_pred):
#     accuracy = 0.0
#     # print("TTTT", y_real.shape)
#     for i in range(y_real.shape[1]):
#         accuracy = custom_accuracy_item(y_real[i], y_pred[i])
#     accuracy = accuracy / y_real.shape[1]
#     return tf.reduce_mean(accuracy)

def custom_accuracy_item(y_real, y_pred):
    condition = tf.math.abs(1 - (y_pred / y_real)) < 0.1
    return tf.cast(condition, tf.float32)

@register_keras_serializable(package="Custom", name="custom_accuracy")
def custom_accuracy(y_real, y_pred):
    accuracies = []
    for i in range(y_real.shape[1]):
        accuracy = custom_accuracy_item(y_real[:, i], y_pred[:, i])
        accuracies.append(tf.reduce_mean(accuracy))
    return tf.reduce_mean(tf.stack(accuracies))

@register_keras_serializable(package="Custom", name="TemperatureMeasurements")
class TemperatureMeasurements:
    #pd.read_csv('./query_result.csv', sep=',')
    def __init__(self, filePath, chosen_category, number_days_to_predict_next, number_days_predict_future):
        self.filePath = filePath
        self.chosen_category = chosen_category
        self.number_days_to_predict_next = number_days_to_predict_next
        self.number_days_predict_future = number_days_predict_future

    def __read_file(self, file):
        csv = pd.read_csv(file, sep=',')
        csv.head()
        return csv
    
        return {
            "filePath": self.filePath,
            "chosen_category": self.chosen_category,
            "number_days_to_predict_next": self.number_days_to_predict_next,
            "number_days_predict_future": self.number_days_predict_future
        }
        
    def get_config(self):
        return {
            "filePath": self.filePath,
            "chosen_category": self.chosen_category,
            "number_days_to_predict_next": self.number_days_to_predict_next,
            "number_days_predict_future": self.number_days_predict_future
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    #Função para interpolar uma série temporal com spline cúbico
    def interpolate_spline(self, series):
        # Remove os valores NaN
        series_nonan = series.dropna()
        if len(series_nonan) < 2:  # Precisamos de pelo menos dois pontos para interpolar
            return series

        # Interpolação spline cúbica
        spline = CubicSpline(series_nonan.index, series_nonan.values)
        return pd.Series(spline(series.index), index=series.index)

    def __normalize_invalid_rows(self, database, index_column_chosen_category):
        print("index_column_chosen_category", index_column_chosen_category)
        df = pd.DataFrame(database, columns=[f'value{i}' for i in range(0, database.shape[1])])
        for col in df.columns[0:]:
            df[col] = self.interpolate_spline(df[col])
        base = df.to_numpy()
        return base

    def __normalize_base(self, base_train):
        normalizer = MinMaxScaler(feature_range=(0,1))
        normalize_base_train = normalizer.fit_transform(base_train)
        normalize_base_train = np.nan_to_num(normalize_base_train)
        joblib.dump(normalizer, 'normalizer.save')
        return normalize_base_train, normalizer

    def __rearrange_array_in_number_of_days(self, number_days_to_predict_next, total_items, normalize_base_train, categories_total, database):
        forecasters = []
        data_forecasters = []
        values_to_predict = []
        for i in range(number_days_to_predict_next, total_items):
            data_forecasters.append(database[i, 0])
            forecasters.append(normalize_base_train[i-number_days_to_predict_next:i, 0:categories_total])
            values_to_predict.append(normalize_base_train[i])
        forecasters, values_to_predict = np.array(forecasters), np.array(values_to_predict)
        return forecasters, values_to_predict, data_forecasters

    def __prediction(self, forecasters_train, forecasters_test, result_train):
        l2_reg = L2(0.01)
        regressor = Sequential()
        regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (forecasters_train.shape[1], forecasters_train.shape[2]), kernel_regularizer=l2_reg))
        regressor.add(LeakyReLU(alpha=0.5))
        regressor.add(LSTM(units = 80, return_sequences = True))
        regressor.add(LeakyReLU(alpha=0.5))
        regressor.add(LSTM(units = 80, return_sequences = True))
        regressor.add(LeakyReLU(alpha=0.5))
        regressor.add(LSTM(units = 80))
        regressor.add(LeakyReLU(alpha=0.5))
        regressor.add(Dense(units = result_train.shape[1], activation = 'sigmoid'))
        optimizer = Adam(learning_rate=0.01)
        regressor.compile(optimizer = optimizer, loss = 'mean_squared_error',
                        metrics = ['mean_absolute_error', custom_accuracy])
        es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
        rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
        mcp = ModelCheckpoint(filepath = 'pesos.keras', monitor = 'loss', save_best_only = True, verbose = 1)
        regressor.fit(forecasters_train, result_train, epochs = 100, batch_size = 32, callbacks = [es, rlr, mcp])
        predictions = regressor.predict(forecasters_test)
        # regressor.save("./models/pesos.keras")

        return predictions, regressor

    def __show_results_prediction(self, result_test, forecasters_train, predictions, normalizer, base_array, data_tests, index_column_chosen_category):
        base_prediction = np.zeros((result_test.shape[0], result_test.shape[1]))        
        prediction_original_format = normalizer.inverse_transform(base_prediction)
        result_test_original = normalizer.inverse_transform(result_test)
        predictions_original = normalizer.inverse_transform(predictions)

        for i in range(len(prediction_original_format)):
            item = result_test_original[i]
            for j in range(len(item)):
                print(f"{data_tests[i]} - Esperado: {"{:.2f}".format(result_test_original[i][j])}, Previsão: {"{:.2f}".format(predictions_original[i][j])}")
            print("########################")

    def __future_prediction(self, future_forecasters):
        current_path = os.getcwd() + '/pesos.keras'
        custom_objects = {
            'custom_accuracy': custom_accuracy
        }
        regressor = load_model(current_path, custom_objects=custom_objects)
        future_predictions = regressor.predict(future_forecasters)
        return future_predictions
    
    # def format_expected_days_with_data(self, expected_days, database):
    #     newPrevisions = []
    #     last_date = np.array(database[database.shape[0]-1])[0]
    #     converted_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')
    #     print(last_date)
    #     for prevision in expected_days:
    #         converted_date = converted_date + timedelta(hours=1)
    #         converted_date_text = converted_date.strftime("%d-%m-%Y %H:%M:%S")
    #         array = np.insert(prevision, 0, converted_date_text)
    #         newPrevisions.append(array)
    #     return newPrevisions
    
    def format_expected_days_with_data(self, expected_days, database):
        newDates = []
        last_date = np.array(database[database.shape[0]-1])[0]
        converted_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')
        print(last_date)
        for prevision in expected_days:
            converted_date = converted_date + timedelta(hours=1)
            converted_date_text = converted_date.strftime("%d-%m-%Y %H:%M:%S")
            newDates.append(converted_date_text)
        return newDates

            
    def getFuturePredictions(self):
        csv = self.__read_file(self.filePath)
        csv.replace(-9999.0, np.nan, inplace=True)
        csv.replace(-9999, np.nan, inplace=True)
        database = csv.to_numpy()
        categories = csv.columns.tolist()
        index_column_chosen_category = categories.index(self.chosen_category)
        base_without_date = database[:, 1:database.shape[1]]
        base_array = self.__normalize_invalid_rows(base_without_date, index_column_chosen_category)
        base_train = base_array
        normalize_base_train, normalizer = self.__normalize_base(base_train)
        forecasters = []
        data_forecasters = []
        values_to_predict = []
        total_items = normalize_base_train.shape[0]
        categories_total = normalize_base_train.shape[1]
        forecasters, values_to_predict, data_forecasters = self.__rearrange_array_in_number_of_days(self.number_days_to_predict_next, total_items, normalize_base_train, categories_total, database)

        print("FOFOFOOFOF", forecasters[forecasters.shape[0]-1])

        forecasters_train, forecasters_test, result_train, result_test = train_test_split(forecasters, values_to_predict, test_size=0.3, shuffle=False)
        index_init_result_test = len(forecasters_train)
        data_tests = data_forecasters[index_init_result_test:]
        normalize_base_train.shape
        predictions, regressor = self.__prediction(forecasters_train, forecasters_test, result_train)
        self.__show_results_prediction(result_test, forecasters_train, predictions, normalizer, base_array, data_tests, index_column_chosen_category)

        #   print("$$$", self.number_days_to_predict_next * 24) # 144
        # total_items_with_predict_values = total_items + (self.number_days_to_predict_next * 24)
        # print(total_items, total_items_with_predict_values) # 14400 14544
        # new_rows = np.zeros((total_items_with_predict_values - database.shape[0], database.shape[1]))
        # new_database = np.vstack((database, new_rows))
        # new_rows_normalize_base_train = np.zeros((total_items_with_predict_values - normalize_base_train.shape[0], normalize_base_train.shape[1]))
        # new_normalize_base_train = np.vstack((normalize_base_train, new_rows_normalize_base_train))
        # future_forecasters, future_values_to_predict, future_data_forecasters = self.__rearrange_array_in_number_of_days(self.number_days_to_predict_next, total_items_with_predict_values, new_normalize_base_train, categories_total, new_database)
        
        # print("ABACAXI", future_forecasters.shape)
        
        # future = self.__future_prediction(future_forecasters)
        # future_prediction_original_format = normalizer.inverse_transform(future)
        # expected_days = future_prediction_original_format[self.number_days_predict_future:]
        # prediction_dates = self.format_expected_days_with_data(expected_days, database)
        # return expected_days, prediction_dates
        