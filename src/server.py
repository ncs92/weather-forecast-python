import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qsl, urlparse
from temperature_measurements import TemperatureMeasurements
from future_predictions import FuturePredictions
from datetime import datetime

# https://realpython.com/python-http-server/
class RequestHandler(BaseHTTPRequestHandler):


  def format_response(self, temperatures, prediction_dates):
    response = []
    i = 0
    for temperature in temperatures:
        response.append({
            'date': prediction_dates[i], 
            'precipitation': float(temperature[0]), 
            'atmospheric_pressure_season_level_hourly': float(temperature[1]), 
            'max_atmospheric_pressure_previous_hour': float(temperature[2]), 
            'atmospheric_pressure_min_earliest_hour': float(temperature[3]), 
            'global_radiation': float(temperature[4]), 
            'air_temperature_dry_bulb_hours': float(temperature[5]), 
            'dew_point_temperature': float(temperature[6]), 
            'maximum_temperature_previous_hour': float(temperature[7]), 
            'minimum_temperature_previous_hour': float(temperature[8]), 
            'dew_temperature_max_earliest_hour': float(temperature[9]), 
            'dew_temperature_min_earliest_hour': float(temperature[10]), 
            'humidity_rel_max_earliest_hour': float(temperature[11]), 
            'humidity_rel_min_earliest_hour': float(temperature[12]), 
            'relative_air_humidity_hourly': float(temperature[13]), 
            'wind_clockwise_direction': float(temperature[14]), 
            'wind_maximum_gust': float(temperature[15]), 
            'wind_hourly_speed': float(temperature[16])
        })
        i = i + 1
    return response
        

  def do_GET(self):
    url = urlparse(self.path)
    query = dict(parse_qsl(url.query))
    
    if url.path == "/weather-forecast":
        # temperatureMeasurements = TemperatureMeasurements('./src/query_result.csv', 'maximum_temperature_previous_hour', 6, 1)
        filePath='/pesos.keras'
        number_days_predict_future=6
        data = [0.07778849, 0.76898223, 0.45447155, 0.42938031, 0.86328745, 0.92458115,
        0.1472045, 0.93643693, 0.08898623, 0.96877748, 0.98860011, 0.3510132,
        0.10678067, 0.04347153, 0.06222543, 0.95923277, 0.97397048]
        initialDate = '2024-07-31 11:00:00'
        forecasters = FuturePredictions(filePath, number_days_predict_future, data, initialDate)
        temperatures, prediction_dates = forecasters.getFutureForecasts()
        data_hora_atual = datetime.now()
        data_formatada = data_hora_atual.strftime("%d-%m-%Y %H:%M:%S")
        response = json.dumps({
          "previsoes": self.format_response(temperatures, prediction_dates),
          "data_solicitação": data_formatada
        })
    else:
        response = json.dumps({ "path": url.path, "query": query })

    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(response.encode("utf-8"))

if __name__ == "__main__":
  host = "localhost"
  port = 3011
  print(f"Server started at http://{host}:{port}")
  server = HTTPServer((host, port), RequestHandler)
  server.timeout = 2400
  server.serve_forever()
