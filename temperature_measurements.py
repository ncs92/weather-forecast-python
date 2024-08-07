import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.regularizers import L1L2, L1, L2
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import math

csv = pd.read_csv('./query_result.csv', sep=',')

csv

csv.head()

database = csv.to_numpy()
database.shape

database[0:1]

categories = csv.columns.tolist()

categories

chosen_category = 'precipitation'
index_column_chosen_category = categories.index(chosen_category)
index_column_chosen_category

list_without_invalid_rows = [item for item in database if (not math.isnan(item[index_column_chosen_category])) and item[index_column_chosen_category] != -9999]
len(list_without_invalid_rows)

list_without_invalid_rows[:5]

base_array = np.array(list_without_invalid_rows)
base_train = base_array[:, 1:18]
base_train[:5]

base_train.shape

normalizer = MinMaxScaler(feature_range=(0,1))
normalize_base_train = normalizer.fit_transform(base_train)
normalize_base_train[:5]

normalize_base_train.shape

normalize_base_train = np.nan_to_num(normalize_base_train)
normalize_base_train[:5]

normalize_base_train.shape



forecasters = []
data_forecasters = []
values_to_predict = []
total_items = normalize_base_train.shape[0]
number_days_to_predict_next = 6
categories_total = normalize_base_train.shape[1]

for i in range(number_days_to_predict_next, total_items):
    data_forecasters.append(base_array[i, 0])
    forecasters.append(normalize_base_train[i-number_days_to_predict_next:i, 0:categories_total])
    values_to_predict.append(normalize_base_train[i, index_column_chosen_category])
forecasters, values_to_predict = np.array(forecasters), np.array(values_to_predict)

values_to_predict[:5]

forecasters_train, forecasters_test, result_train, result_test = train_test_split(forecasters, values_to_predict, test_size=0.3, shuffle=False)

len(data_forecasters)

index_init_result_test = len(forecasters_train)
data_tests = data_forecasters[index_init_result_test:]

print(len(data_tests[index_init_result_test:]), len(forecasters_train), len(result_test))

normalize_base_train.shape

def custom_accuracy(y_real, y_pred):
    # A fórmula é |1 - (y_pred / y_true)| < 0.1, retornando um tensor booleano
    condition = tf.math.abs(1 - (y_pred / y_real)) < 0.1
    # Convertendo o tensor booleano em float (True se torna 1.0, False se torna 0.0)
    as_float = tf.cast(condition, tf.float32)
    # Calculando a média para obter a acurácia
    return tf.reduce_mean(as_float)

forecasters_train.shape

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

regressor.add(Dense(units = 1, activation = 'sigmoid'))

optimizer = Adam(learning_rate=0.001)
regressor.compile(optimizer = optimizer, loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error', custom_accuracy])
es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.keras', monitor = 'loss', save_best_only = True, verbose = 1)
regressor.fit(forecasters_train, result_train, epochs = 100, batch_size = 32, callbacks = [es, rlr, mcp])
predictions = regressor.predict(forecasters_test)

predictions.mean()

result_test.mean()

result_test.shape

base_prediction = np.zeros((result_test.shape[0], forecasters_train.shape[2]))
base_prediction[:, 4] = predictions[:, 0]
prediction_original_format = normalizer.inverse_transform(base_prediction)
#result_test_original_format = normalizer.inverse_transform(result_test) #.reshape(-1, 1))
base_prediction[:1]

predicoes = prediction_original_format[:, 4]
predicoes[:5]

result_test.shape

base_zeros = np.zeros((result_test.shape[0], forecasters_train.shape[2]))
base_zeros[:, 4] = result_test
result_test_original_format = normalizer.inverse_transform(base_zeros)

result_test_original = result_test_original_format[:, 4]
result_test_original[:5]

base_train[0]

result_test_original[0]

data = base_array

for i in range(len(predicoes)):
    print(f"{data_tests[i]} - Esperado: {result_test_original[i]}, Previsão: {predicoes[i]}")

intervalo = 80  # Por exemplo, mostrando uma data a cada 7 dias
datas_filtradas = data_tests[::intervalo]  # Selecionando datas em intervalos
indices_filtrados = range(0, len(data_tests), intervalo)

plt.figure(figsize=(10, 6))
plt.plot(result_test_original, color = 'red', label = 'Temperatura média')
plt.plot(predicoes, color = 'blue', label = 'Previsões')
plt.title('Previsão da temperatura média de São Paulo')
plt.xlabel('Tempo')
plt.ylabel('Temperatura')

date_format = mdates.DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.xticks(ticks=indices_filtrados, labels=datas_filtradas, rotation=45)
plt.legend()
plt.show()

df = pd.DataFrame(base_array, columns=categories)
df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
correlacao = df[categories].corr()
correlacao



import numpy as np
from datetime import datetime, timedelta

previsoes = []
ultimos_dias = forecasters_test
pos = len(data_tests)-1
data_inicial_str = data_tests[pos]
data_inicial = datetime.strptime(data_inicial_str, '%Y-%m-%d %H:%M:%S')
datas = []
number_days = 60

for _ in range(number_days):  # Prever os próximos 5 dias
    nova_data = data_inicial + timedelta(days=i)
    datas.append(nova_data.strftime('%Y-%m-%d %H:%M:%S'))
    previsao_atual = regressor.predict(ultimos_dias)
    nova_entrada = np.zeros(17)
    nova_entrada[4] = previsao_atual.flatten()[0]
    #previsoes.append(previsao_atual.flatten()[0])
    previsoes.append(nova_entrada)
    ultimos_dias = np.array([np.vstack([item[1:], nova_entrada]) for item in ultimos_dias])

previsoes_array = np.array(previsoes)

previsoes_array_original_format = normalizer.inverse_transform(previsoes_array)

previsoes_array_original = previsoes_array_original_format[:, 4]

# Exibir as previsões transformadas
for i, previsao in enumerate(previsoes_array_original):
    print(f"Previsão para o dia {datas[i]}: {previsao}")

previsoes_array.shape
