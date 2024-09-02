from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import keras
import joblib

def custom_accuracy_item(y_real, y_pred):
    # A fórmula é |1 - (y_pred / y_true)| < 0.1, retornando um tensor booleano
    # (y_pred * 1) / y_real = x
    condition = tf.math.abs(1 - (y_pred / y_real)) < 0.1
    # Convertendo o tensor booleano em float (True se torna 1.0, False se torna 0.0)
    as_float = tf.cast(condition, tf.float32)
    # Calculando a média para obter a acurácia
    return as_float

def custom_accuracy(y_real, y_pred):
    accuracy = 0.0
    # print("TTTT", y_real.shape)
    for i in range(y_real.shape[1]):
        accuracy = custom_accuracy_item(y_real[i], y_pred[i])
    accuracy = accuracy / y_real.shape[1]
    return tf.reduce_mean(accuracy)

class FuturePredictions:
    def __init__(self, filePath, number_days_predict_future, data, initialDate):
        self.filePath = filePath
        self.number_days_predict_future = number_days_predict_future
        self.data = data
        self.total_number_hours = self.number_days_predict_future * 24
        self.total_items = self.number_days_predict_future * 24
        self.initialDate = initialDate
        
    def __rearrange_array_in_number_of_days(self, database, categories_total):
        forecasters = []
        values_to_predict = []
        print("FFFF", self.number_days_predict_future)
        print("DDDD", self.total_items)
        for i in range(self.number_days_predict_future, self.total_items):
            forecasters.append(database[i-self.number_days_predict_future:i, 0:categories_total])
            values_to_predict.append(database[i])
        forecasters, values_to_predict = np.array(forecasters), np.array(values_to_predict)
        return forecasters, values_to_predict
    
    def __future_prediction(self, future_forecasters):
        current_path = os.getcwd() + self.filePath
        custom_objects = {
            'custom_accuracy': custom_accuracy
        }
        regressor = load_model(current_path, custom_objects=custom_objects)
        future_predictions = regressor.predict(future_forecasters)
        return future_predictions
    
    def __format_expected_days_with_data(self, database):
        newDates = []
        last_date = self.initialDate
        converted_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')
        for prevision in database:
            converted_date = converted_date + timedelta(hours=1)
            converted_date_text = converted_date.strftime("%d-%m-%Y %H:%M:%S")
            newDates.append(converted_date_text)
        return newDates
    
    def getFutureForecasts(self):
        categories_total = len(self.data)
        new_rows = np.zeros((self.total_number_hours, categories_total))
        new_database = np.vstack((self.data, new_rows))
        self.total_items = new_database.shape[0]
        forecasters, values_to_predict = self.__rearrange_array_in_number_of_days(new_database, categories_total)    
        
        future = self.__future_prediction(forecasters)
        normalizer = joblib.load('normalizer.save')
        
        future_prediction_original_format = normalizer.inverse_transform(future)
        expected_days = future_prediction_original_format[self.number_days_predict_future:]
        prediction_dates = self.__format_expected_days_with_data(new_database)
        return expected_days, prediction_dates