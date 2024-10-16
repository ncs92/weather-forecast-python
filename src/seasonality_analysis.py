import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import os

class SeasonalityAnalysis:
    def __init__(self, baseDir, data):
        self.baseDir = baseDir
        self.data = data
    
    def data_decomposition(self):
        print("$$$$$$$")
        print(self.data['historical'])
        print("$$$$$$$")
        print("######")
        print(self.data['forecast'])
        print("$$$$$$$")
        period = len(self.data['historical'])
        self.historical_decomposition = sm.tsa.seasonal_decompose(self.data['historical'], model='additive', period=7)
        self.decomposition_forecast = sm.tsa.seasonal_decompose(self.data['forecast'], model='additive', period=7)

        # Plotando a decomposição
        self.historical_decomposition.plot()
        plt.title('Decomposição dos Dados Históricos')
        file_path = os.path.join(self.baseDir, 'historical_decomposition.png')
        plt.savefig(file_path)
        # plt.show()

        self.decomposition_forecast.plot()
        plt.title('Decomposição dos Dados Previstos')
        file_path = os.path.join(self.baseDir, 'forecast_decomposition.png')
        plt.savefig(file_path)
        # plt.show()
        
    def comparison_trends_seasonalities(self):
        # Plotando a tendência
        plt.figure(figsize=(10, 6))
        plt.plot(self.historical_decomposition.trend, label='Tendência Histórica')
        plt.plot(self.decomposition_forecast.trend, label='Tendência Prevista')
        plt.legend(loc='upper left')
        plt.title('Comparação das Tendências')
        file_path = os.path.join(self.baseDir, 'tendencies.png')
        plt.savefig(file_path)
        # plt.show()

        # Plotando a sazonalidade
        plt.figure(figsize=(10, 6))
        plt.plot(self.historical_decomposition.seasonal, label='Sazonalidade Histórica')
        plt.plot(self.decomposition_forecast.seasonal, label='Sazonalidade Prevista')
        plt.legend(loc='upper left')
        plt.title('Comparação das Sazonalidades')
        file_path = os.path.join(self.baseDir, 'seasonality.png')
        plt.savefig(file_path)
        # plt.show()

    def pearson_correlation(self):
        # Calculando correlação de Pearson entre as tendências
        correlacao_tendencia = np.corrcoef(self.historical_decomposition.trend, self.decomposition_forecast.trend)[0, 1]
        print(f'Correlação entre tendências: {correlacao_tendencia}')

        # Calculando correlação de Pearson entre as sazonalidades
        correlacao_sazonalidade = np.corrcoef(self.historical_decomposition.seasonal, self.decomposition_forecast.seasonal)[0, 1]
        print(f'Correlação entre sazonalidades: {correlacao_sazonalidade}')
        
        return correlacao_tendencia, correlacao_sazonalidade
