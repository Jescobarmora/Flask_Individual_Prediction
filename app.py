from flask import Flask, request, jsonify
import json
import pandas as pd
import pickle
import traceback

# Definir el path de tu modelo y el archivo JSON donde se guardarán las predicciones
path = "/code/Python/Corte_2/Quiz_2_2/Punto_2/"
file_name = 'predictions/predictions.json'

# Crear una instancia de Flask
app = Flask(__name__)

# Cargar el modelo preentrenado desde el archivo pickle
with open(path + 'models/ridge_model.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
    try:
        with open(path + file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    predictions.append(prediction_data)

    with open(path + file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

# Página de inicio
@app.route("/", methods=["GET"])
def home():
    return "Predicción estudiantes"

# Endpoint para realizar la predicción
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recibir los datos en formato JSON
        data = request.json

        # Crear DataFrame a partir de los datos de entrada
        user_data = pd.DataFrame([data])

        # Renombrar las columnas para que coincidan con lo que el modelo espera
        user_data.rename(columns={
            'Dominio': 'dominio',
            'Tecnologia': 'Tec',
            'Avg_Session_Length': 'Avg. Session Length',
            'Time_on_App': 'Time on App',
            'Time_on_Website': 'Time on Website',
            'Length_of_Membership': 'Length of Membership'
        }, inplace=True)

        # Realizar predicción
        yhat = modelo.predict(user_data)

        # Guardar predicción con ID en el archivo JSON
        prediction_result = {
            "Email": user_data["Email"].values[0], 
            "Prediction": yhat[0]
        }
        save_prediction(prediction_result)

        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}", "traceback": traceback.format_exc()}), 500

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
