from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import keras
import joblib

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
    def __init__(self, last_forecast_file_path, last_predict_value_file_path, model_path, normalizer_path, last_date, number_predictions, number_days_predict_future):
        self.forecast = np.loadtxt(last_forecast_file_path)
        print(f"FOOOOOR forecasters: {self.forecast}")
        print(f"INITIAL DATE: {last_date}")
        print(f"number_predictions : {number_predictions}")
        self.predict_value = np.loadtxt(last_predict_value_file_path)
        self.initial_date = last_date
        self.number_predictions = number_predictions
        self.model_path = model_path
        self.number_days_predict_future = number_days_predict_future
        self.normalizer_path = normalizer_path

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
        print("initial ", self.initial_date)
        converted_date = datetime.strptime(last_date, '%Y-%m-%dT%H:%M:%S.%fZ')
        for prevision in database:
            converted_date = converted_date + timedelta(hours=1)
            converted_date_text = converted_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            newDates.append(converted_date_text)
        return newDates

    def getFutureForecasts(self):
        print("MOOOOB")
        print(self.forecast.shape)
        categories_total = self.forecast.shape[1]
        new_rows = np.zeros((self.number_predictions, categories_total))
        new_database = np.vstack((self.forecast, new_rows))
        self.total_items = new_database.shape[0]
        print(f"new_database: {new_database.shape}")
        converted_forecast = np.vectorize(lambda x: float(x))(self.forecast)
        print(f"converted_forecast: {converted_forecast.shape}")
        last_data  = converted_forecast[-6:]
        print(f"last_data: {last_data.shape}")
        data = last_data.reshape(1, self.number_days_predict_future , categories_total)
        print("############")
        print(data)
        all_predictions = data

        for i in range(self.number_predictions):
            future = self.__future_prediction(data)
            print(f"FUTURE: {future}")
            data = np.delete(data, 0, axis=1)
            data = np.append(data, future[np.newaxis, :], axis=1)
            resultado = np.vstack([all_predictions[0], future])
            all_predictions = np.array([resultado])
            
            print("############")
            print(data)
        
        # print("###########")
        # print(f"categories_total: {categories_total}")
        # print(f"new_rows: {new_rows.shape}")
        # print(f"new_database: {new_database.shape}") # (43, 6)
        # print(f"total_items: {self.total_items}")
        # print("###########")
        # print(new_database[:1])
        
        # print(f"forecasters: {forecasters}") # (37, 6, 6)
        # print(forecasters[:1])
        #(132085, 6, 1)
        print("normalizer_path ", self.normalizer_path)
        normalizer = joblib.load(self.normalizer_path)
        # print("###########")
        # print(f"normalizer_path: {normalizer_path}")
        # print("###########")
        print(f"data: {all_predictions}")
        # values = data.reshape(-1, 1)
        # print("valuuuuuues", values)
        data_reshaped = all_predictions.reshape(-1, all_predictions.shape[-1])
        future_prediction_original_format = normalizer.inverse_transform(data_reshaped)
        
        print("###########")
        print(f"future_prediction_original_format: {future_prediction_original_format}")
        print(f"future_prediction_original_format: {future_prediction_original_format.shape}")
        
        prediction_dates = self.__format_expected_days_with_data(future_prediction_original_format)
        print("###########")
        print(f"prediction_dates: {prediction_dates}")
        print("###########")
        # format_values = future_prediction_original_format.flatten().tolist()
        return future_prediction_original_format[:, -1], prediction_dates