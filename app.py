from flask import Flask, jsonify, render_template, request
import logging
import joblib
import pandas as pd

app = Flask(__name__)
# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('model.pkl')
app.logger.debug('Modelo cargado correctamente.')
#categorias de las enfermedades
categoria = {
    0: 'un estado Normal',
    1: 'una Enfermedad cardíaca'
}

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #obtener datos 
        MaxHR = float(request.form['MaxHR'])
        ST_Slope = float(request.form['ST_Slope'])
        Oldpeak = float(request.form['Oldpeak'])
        Cholesterol = float(request.form['Cholesterol'])

        data_df = pd.DataFrame([[MaxHR,ST_Slope,Oldpeak,Cholesterol]],columns =['MaxHR','ST_Slope','Oldpeak','Cholesterol'])
        app.logger.debug(f'DataFrame creado: {data_df}')

           # Realizar predicciones
        prediction = model.predict(data_df)
        predicted_category = categoria.get(prediction[0],"categoria desconocida")  # Convertir a lista para serializar a JSON
        app.logger.debug(f'Predicción: {predicted_category}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': predicted_category})
        
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
