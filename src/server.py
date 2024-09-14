import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qsl, urlparse
from temperature_measurements import TemperatureMeasurements
from future_predictions import FuturePredictions
from datetime import datetime
import json

class RequestHandler(BaseHTTPRequestHandler):  
  def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = post_data.decode('utf-8')
        try:
          post_data_json = json.loads(post_data)
          print("###########")
          print(f"Dados recebidos: {post_data_json}")
          print("###########")
          if url.path == "/weather-forecast":
            temperatureMeasurements = TemperatureMeasurements(post_data_json["filePath"], post_data_json["baseDir"], post_data_json["forecast"])
            response = temperatureMeasurements.getFuturePredictions()
              # Definir o código de status de resposta
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
          if url.path == "/predict-weather-forecast":
            last_forecast_file_path = post_data_json["last_forecast_file_path"]
            last_predict_value_file_path = post_data_json["last_predict_value_file_path"]
            model_path = post_data_json["model_path"]
            last_date = post_data_json["last_date"]
            number_predictions = post_data_json["number_predictions"]
            normalizer_path = post_data_json["normalizer_path"]
            number_days_predict_future = post_data_json["number_days_predict_future" ]
            futurePredictions = FuturePredictions(last_forecast_file_path, last_predict_value_file_path, model_path, normalizer_path, last_date, number_predictions, number_days_predict_future)
            predictions, days = futurePredictions.getFutureForecasts()
            print("######Predictions######", len(predictions))
            print(predictions)
            print("######Predictions######", len(days))
            json_array = []
            for i in range(len(predictions) - 1):  # Mude o range para o número de itens que deseja
              data = {
                  "date": days[i],
                  "value": predictions[i]
              }
              json_array.append(data)
            json_output = json.dumps(json_array, indent=4)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json_output.encode("utf-8"))
            
        except json.JSONDecodeError:
          print(f"Erro ao decodificar JSON: {post_data}")

if __name__ == "__main__":
  host = "localhost"
  port = 3011
  print(f"Server started at http://{host}:{port}")
  server = HTTPServer((host, port), RequestHandler)
  server.timeout = 2400
  server.serve_forever()
