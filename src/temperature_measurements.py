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
    def __init__(self, filePath, baseDir, request):
        print(request)
        self.filePath = filePath
        self.baseDir = baseDir
        self.chosen_category = request["category"]
        self.number_days_to_predict_next = request["quantity_predict_next"]
        self.epochs = request["epochs"]
        self.batch_size = request["batch_size"]
        self.kernel_regularizer = request["kernel_regularizers"]["type"]
        self.layer_weight_l1 = request["kernel_regularizers"]["layer_weight_l1"]
        self.layer_weight_l2 = request["kernel_regularizers"]["layer_weight_l2"]
        self.inner_layers = request["inner_layers"]
        self.learning_rate = float(request["learning_rate"])
        # self.number_days_predict_future = number_days_predict_future

    def __read_file(self, file):
        csv = pd.read_csv(file, sep=',')
        csv.head()
        return csv

        return {
            "filePath": self.filePath,
            "chosen_category": self.chosen_category,
            "number_days_to_predict_next": self.number_days_to_predict_next,
            # "number_days_predict_future": self.number_days_predict_future
        }

    def get_config(self):
        return {
            "filePath": self.filePath,
            "chosen_category": self.chosen_category,
            "number_days_to_predict_next": self.number_days_to_predict_next,
            # "number_days_predict_future": self.number_days_predict_future
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def interpolate_spline(self, series):
        print("SERRR", series)
        series_nonan = series.dropna()
        print("series_nonan", series_nonan)
        if len(series_nonan) < 2:
            return series
        spline = CubicSpline(series_nonan.index, series_nonan.values)
        print("spline", spline)
        result = pd.Series(spline(series.index), index=series.index)
        print("result", result)
        return result

    def __normalize_invalid_rows(self, database, index_column_chosen_category):
        df = pd.DataFrame(database, columns=[f'value{i}' for i in range(0, database.shape[1])])
        for col in df.columns[0:]:
            df[col] = self.interpolate_spline(df[col])
        base = df.to_numpy()
        return base

    def __normalize_base(self, base_train):
        normalizer = MinMaxScaler(feature_range=(0,1))
        normalize_base_train = normalizer.fit_transform(base_train)
        normalize_base_train = np.nan_to_num(normalize_base_train)

        normalizer_path = os.path.join(self.baseDir, 'normalizer.save')
        if os.path.exists(normalizer_path):
            os.remove(normalizer_path)

        joblib.dump(normalizer, normalizer_path)
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

    def __prediction(self, forecasters_train, forecasters_test, result_train, filepath, model_file_path):
        reg = L2(self.layer_weight_l2)
        if (self.kernel_regularizer == 'L1'):
            reg = L1(self.layer_weight_l1)
        elif (self.kernel_regularizer == 'L1L2'):
            reg = L1L2(l1=self.layer_weight_l1, l2=self.layer_weight_l2)
        regressor = Sequential()
        regressor.save_weights(filepath)
        regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (forecasters_train.shape[1], forecasters_train.shape[2]), kernel_regularizer=L2(0.001)))
        # regressor.add(LeakyReLU(alpha=0.5))
        cont = 0
        for innerLayer in self.inner_layers:
            if cont < (len(self.inner_layers)-1):
                regressor.add(LSTM(units = innerLayer["number_units"], kernel_regularizer=reg, return_sequences = True))
            else:
                regressor.add(LSTM(units = innerLayer["number_units"], kernel_regularizer=reg))
            if (innerLayer["use_leaky_relu"]):
                regressor.add(LeakyReLU(alpha=innerLayer["relu_alpha"]))
            cont = cont + 1

        regressor.add(Dense(units = result_train.shape[1], activation = 'sigmoid'))
        optimizer = Adam(learning_rate=self.learning_rate) #
        regressor.compile(optimizer = optimizer, loss = 'mean_squared_error',
                        metrics = ['mean_absolute_error', custom_accuracy])
        es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 5, verbose = 1, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 10, verbose = 1)
        mcp = ModelCheckpoint(filepath = filepath, monitor = 'loss', save_best_only = True, verbose = 1, save_weights_only=True)
        history = regressor.fit(forecasters_train, result_train, epochs = self.epochs, batch_size = self.batch_size, callbacks = [es, rlr, mcp])
        final_metrics = history.history
        predictions = regressor.predict(forecasters_test)
        regressor.save(model_file_path)
        custom_acc = final_metrics['custom_accuracy'][-1]
        loss = final_metrics['loss'][-1]
        mean_absolute_error = final_metrics['mean_absolute_error'][-1]
        learning_rate = regressor.optimizer.learning_rate.numpy()
        print(f"Custom Accuracy: {custom_acc}")
        print(f"Loss: {loss}")
        print(f"Mean Absolute Error: {mean_absolute_error}")
        print(f"Learning Rate: {learning_rate}")
        print("----------------------------")
        print(predictions)

        return predictions, regressor, custom_acc, loss, mean_absolute_error, learning_rate

    def __show_results_prediction(self, result_test, forecasters_train, predictions, normalizer, base_array, data_tests, index_column_chosen_category):
        base_prediction = np.zeros((result_test.shape[0], result_test.shape[1]))
        prediction_original_format = normalizer.inverse_transform(base_prediction)
        result_test_original = normalizer.inverse_transform(result_test)
        predictions_original = normalizer.inverse_transform(predictions)
        predictions = []
        result_test = []

        for i in range(len(prediction_original_format)):
            item = result_test_original[i]
            tam = 10 if len(item) > 10 else len(item)
            for j in range(tam):
                predictions.append(predictions_original[i][j])
                result_test.append(result_test_original[i][j])
                print(f"{data_tests[i]} - Esperado: {"{:.2f}".format(result_test_original[i][j])}, Previsão: {"{:.2f}".format(predictions_original[i][j])}")
            print("########################")
            
        predictions_path = os.path.join(self.baseDir, 'predictions.txt')
        result_test_file_path = os.path.join(self.baseDir, 'result_test.txt')
        np.savetxt(predictions_path, predictions)
        np.savetxt(result_test_file_path, result_test)

    def __future_prediction(self, future_forecasters):
        current_path = os.getcwd() + '/pesos.keras'
        custom_objects = {
            'custom_accuracy': custom_accuracy
        }
        regressor = load_model(current_path, custom_objects=custom_objects)
        future_predictions = regressor.predict(future_forecasters)
        return future_predictions

    def format_expected_days_with_data(self, expected_days, database):
        newDates = []
        last_date = np.array(database[database.shape[0]-1])[0]
        converted_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')
        for prevision in expected_days:
            converted_date = converted_date + timedelta(hours=1)
            converted_date_text = converted_date.strftime("%d-%m-%Y %H:%M:%S")
            newDates.append(converted_date_text)
        return newDates

    def getOnlyCorrelationColumns(self, csv, chosen_category):
        data_without_date = csv.drop(columns=['date'])
        correlation_matrix = data_without_date.corr()
        chosen_category_corr = correlation_matrix[chosen_category]
        correlated_columns = chosen_category_corr[chosen_category_corr > 0.4].index.tolist()
        print("CORRELATED COLUMNS", correlated_columns)
        if chosen_category not in correlated_columns:
            correlated_columns.append(chosen_category)
        filtered_csv = csv[['date'] + correlated_columns]
        columns = [col for col in filtered_csv.columns if col != chosen_category] + [chosen_category]
        filtered_csv = filtered_csv[columns]

        return filtered_csv


    def getFuturePredictions(self):
        csv = self.__read_file(self.filePath)
        csv.replace(-9999.0, np.nan, inplace=True)
        csv.replace(-9999, np.nan, inplace=True)
        csv.replace([-9999.00, -9999], np.nan, inplace=True)

        filtered_csv = self.getOnlyCorrelationColumns(csv, self.chosen_category)

        database = filtered_csv.to_numpy()
        categories = filtered_csv.columns.tolist()
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

        last_forecast_file_path = os.path.join(self.baseDir, 'forecast.txt')
        last_predict_value_file_path = os.path.join(self.baseDir, 'predict_value.txt')
        if os.path.exists(last_forecast_file_path):
            os.remove(last_forecast_file_path)
        if os.path.exists(last_predict_value_file_path):
            os.remove(last_predict_value_file_path)
        np.savetxt(last_forecast_file_path, forecasters[len(forecasters)-1])
        np.savetxt(last_predict_value_file_path, values_to_predict[len(forecasters)-1])

        forecasters_train, forecasters_test, result_train, result_test = train_test_split(forecasters, values_to_predict, test_size=0.1, shuffle=False)
        index_init_result_test = len(forecasters_train)
        data_tests = data_forecasters[index_init_result_test:]
        normalize_base_train.shape

        file_path = os.path.join(self.baseDir, 'pesos.weights.h5')
        model_file_path = os.path.join(self.baseDir, 'model.h5')
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(model_file_path):
            os.remove(model_file_path)
        predictions, regressor, custom_accuracy, loss, mean_absolute_error, learning_rate = self.__prediction(forecasters_train, forecasters_test, result_train, file_path, model_file_path)
        self.__show_results_prediction(result_test, forecasters_train, predictions, normalizer, base_array, data_tests, index_column_chosen_category)

    
        return {
            'model_file_path': model_file_path,
            'path': file_path,
            'custom_accuracy': str(custom_accuracy),
            'loss': str(loss),
            'mean_absolute_error': str(mean_absolute_error),
            'learning_rate': str(learning_rate)
        }
