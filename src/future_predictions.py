from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import keras
import joblib
import pandas as pd

def custom_accuracy_item(y_real, y_pred):
    condition = tf.math.abs(1 - (y_pred / y_real)) < 0.1
    as_float = tf.cast(condition, tf.float32)
    return as_float

def custom_accuracy(y_real, y_pred):
    accuracy = 0.0
    for i in range(y_real.shape[1]):
        accuracy = custom_accuracy_item(y_real[i], y_pred[i])
    accuracy = accuracy / y_real.shape[1]
    return tf.reduce_mean(accuracy)

class FuturePredictions:
    def __init__(self, last_forecast_file_path, last_predict_value_file_path, model_path, normalizer_path, last_date, number_predictions, number_days_predict_future, typeMeasurement):
        self.forecast = np.loadtxt(last_forecast_file_path)
        self.predict_value = np.loadtxt(last_predict_value_file_path)
        self.initial_date = last_date
        self.number_predictions = number_predictions
        self.model_path = model_path
        self.number_days_predict_future = number_days_predict_future
        self.normalizer_path = normalizer_path
        self.typeMeasurement = typeMeasurement

    def __future_prediction(self, future_forecasters):
        custom_objects = {
            'custom_accuracy': custom_accuracy
        }
        regressor = load_model(self.model_path, custom_objects=custom_objects)
        regressor.compile()
        future_predictions = regressor.predict(future_forecasters)
        return future_predictions
    
    def __format_expected_days_with_data(self, database):
        newDates = []
        last_date = self.initial_date
        converted_date = datetime.strptime(last_date, '%Y-%m-%dT%H:%M:%S.%fZ')
        timeDelta = timedelta(days=1) if self.typeMeasurement == 'day' else timedelta(hours=1)
        for prevision in database:
            converted_date = converted_date + timeDelta
            converted_date_text = converted_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            newDates.append(converted_date_text)
        return newDates

    def getFutureForecasts(self):
        if self.forecast.ndim == 1:
            self.forecast = self.forecast.reshape(-1, 1)
        
        converted_forecast = np.vectorize(lambda x: float(x))(self.forecast)
        categories_total = converted_forecast.shape[1]
        # print("categories_total", categories_total)
        # print("number_predictions", self.number_predictions)
        # new_rows = np.zeros((self.number_predictions, categories_total))
        # print("new_rows", new_rows.shape)
        # new_database = np.vstack((converted_forecast, new_rows))
        # print("new_database", new_database.shape)
        # print("new_database", new_database[10:])
        # self.total_items = new_database.shape[0]
        # print("self.total_items", self.total_items)
        
        # print("converted_forecast", converted_forecast.shape)
        # last_data  = converted_forecast[-6:]
        # print("last_data", last_data.shape)
        data = converted_forecast.reshape(1, converted_forecast.shape[0] , categories_total)
        all_predictions = []
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(all_predictions)

        for i in range(self.number_predictions):
            future = self.__future_prediction(data)
            print(f"FUTURE: {future}")
            all_predictions.append(future)
            data = np.delete(data, 0, axis=1)
            data = np.append(data, future[np.newaxis, :], axis=1)
            # resultado = np.vstack([all_predictions[0], future])
            # all_predictions = np.array([resultado])
            print("---------")
            print(data)
        
        print("$$$$$$$$alll$$$$$$$$$$$$$$$$")
        print(all_predictions)
        numpy_array = np.array(all_predictions)
        print(numpy_array.shape)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        normalizer = joblib.load(self.normalizer_path)
        data_reshaped = numpy_array.reshape(-1, numpy_array.shape[-1])
        future_prediction_original_format = normalizer.inverse_transform(data_reshaped)
        values_string = np.vectorize(lambda x: str(x))(future_prediction_original_format)
        
        prediction_dates = self.__format_expected_days_with_data(future_prediction_original_format)
        return values_string[:, -1], prediction_dates