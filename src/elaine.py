from temperature_measurements import TemperatureMeasurements

temperatureMeasurements = TemperatureMeasurements('./src/query_result.csv', 'maximum_temperature_previous_hour', 6, 3)
temperatures = temperatureMeasurements.getFuturePredictions()
