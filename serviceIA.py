from flask import Flask, request, jsonify, send_file, send_from_directory
from torchvision import transforms
from PIL import Image, ImageDraw
from datetime import datetime
from flask_cors import CORS
from joblib import load
import numpy as np
import base64
import torch
import io
import os

app = Flask(__name__)

CORS(app)

# Carga del modelo
MODEL_PATH = './modelo_deteccion.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Mapeo índice-nombre
index2name = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Effusion",
    3: "Infiltrate",
    4: "Mass",
    5: "Nodule",
    6: "Pneumonia",
    7: "Pneumothorax",
}

# Preprocesamiento de la imagen
def preprocesar_imagen(imagen):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(imagen).unsqueeze(0).to(device)

@app.route('/imagenes_procesadas/<filename>')
def obtener_imagen(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Carpeta para almacenar las imágenes procesadas
UPLOAD_FOLDER = './imagenes_procesadas'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crear la carpeta si no existe

@app.route('/detectar', methods=['POST'])
def detectar():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400
    
    archivo_imagen = request.files['imagen']
    try:
        # Leer y preprocesar la imagen
        imagen = Image.open(io.BytesIO(archivo_imagen.read())).convert('RGB')
        imagen_preprocesada = preprocesar_imagen(imagen)
        
        # Realizar predicción
        with torch.no_grad():
            predictions = model(imagen_preprocesada)
        
        # Extraer las dos cajas con los puntajes más altos
        scores = []
        boxes = []
        names = []
        for i, box in enumerate(predictions[0]["boxes"]):
            score = predictions[0]["scores"][i].cpu().detach().numpy()
            scores.append(score)
        
        # Ordenar por puntaje y seleccionar los dos más altos
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
        for idx in top_indices:
            box = predictions[0]["boxes"][idx]
            boxes.append(box.cpu().tolist())
            
            label = predictions[0]["labels"][idx].item()
            if label >= len(index2name):
                label = 0  # Etiqueta por defecto si el índice es inválido
            name = index2name[label]
            names.append(name)
        
        # Dibujar las cajas sobre la imagen original
        draw = ImageDraw.Draw(imagen)
        for box, name in zip(boxes, names):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # Caja en rojo
            draw.text((x1, y1 - 10), name, fill="red")  # Nombre encima de la caja
        
        # Guardar la imagen en la carpeta
        ruta_imagen = os.path.join(UPLOAD_FOLDER, "imagen_procesada.png")
        ruta_imagen = ruta_imagen.replace("\\", "/") 
        imagen.save(ruta_imagen)
        
        # Preparar respuesta JSON
        respuesta = {
            "prediccion": [{"nombre": names[i], "puntaje": float(scores[top_indices[i]])} for i in range(len(names))],
            "ruta_imagen": ruta_imagen
        }
        
        return jsonify(respuesta)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------- Función para obtener el diagnostico médico de la IA -----------------------
def obtener_diagnostico(nombre_diagnostico):
    # Diccionario de diagnósticos y descripciones
    diagnosticos = {
        "Atelectasis": "Atelactasis: Colapso del tejido pulmonar con pérdida de volumen. Surge cuando los pequeños sacos de aire dentro del pulmón, los alvéolos, pierden aire.",
        "Cardiomegaly": "Cardiomegalia: Aumento del tamaño cardiaco por hipertrofia o dilatación. Suele ser un signo de enfermedad cardiaca o síntoma de otra afección.",
        "Effusion": "Efusion Pleural: Acumulación anormal del líquido entre las capas delgadas del tejido (pleura) que recubre el pulmón y la pared de la cavidad torácica.",
        "Infiltrate": "Infiltrado Pulmonar: Sombras pulmonares anormales, están son el reflejo de una enfermedad o secuela de una enfermedad pulmonar.",
        "Mass": "Tumores Mediastianles: Masas o neoplasias que se forman en el mediastino, una zona en la mitad del tórax que separa los pulmones.",
        "Nodule": "Nodulos Pulmonares: Densidad focal relativamente pequeña en el pulmón. Un nódulo pulmonar solitario (NPS) o lesión en forma de moneda, es una masa en el pulmón de menos de tres centímetros de diámetro.",
        "Pneumonia": "Neumonía: Infección en uno o ambos pulmones. Causa que los alvéolos pulmonares se llenen de líquido o pus.",
        "Pneumothorax": "Neumotórax: Presencia de aire en el espacio pleural que causa colapso pulmonar parcial o completo."
    }
     
    # Devolver el diagnóstico correspondiente o un mensaje de error si no se encuentra
    return diagnosticos.get(nombre_diagnostico, "Diagnóstico no encontrado.")    
    
    
    
#---------------- Devuelve un JSON con la imagen y la prediccion -------------------
@app.route('/detectar2', methods=['POST'])
def detectar2():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400
    
    archivo_imagen = request.files['imagen']
    
    if archivo_imagen.filename == '':
        return "No se seleccionó ningún archivo", 400
    
    # Obtener el nombre original del archivo
    nombre_imagen = archivo_imagen.filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    nombre_imagen = f"{timestamp}_{nombre_imagen}"
    nombre_imagen = nombre_imagen.replace(" ", "_") 
    diagnostico = ""
    
    try:
        # Leer y preprocesar la imagen
        imagen = Image.open(io.BytesIO(archivo_imagen.read())).convert('RGB')
        imagen_preprocesada = preprocesar_imagen(imagen)
        
        # Realizar predicción
        with torch.no_grad():
            predictions = model(imagen_preprocesada)
        
        # Extraer las dos cajas con los puntajes más altos
        scores = []
        boxes = []
        names = []
        for i, box in enumerate(predictions[0]["boxes"]):
            score = predictions[0]["scores"][i].cpu().detach().numpy()
            scores.append(score)
        
        # Ordenar por puntaje y seleccionar los dos más altos
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
        for idx in top_indices:
            box = predictions[0]["boxes"][idx]
            boxes.append(box.cpu().tolist())
            
            label = predictions[0]["labels"][idx].item()
            if label >= len(index2name):
                label = 0  # Etiqueta por defecto si el índice es inválido
            name = index2name[label]
            names.append(name)
            # Obtener el diagnostico de la imagen
            diagnostico = obtener_diagnostico(name)
        
        # Dibujar las cajas sobre la imagen original
        draw = ImageDraw.Draw(imagen)
        for box, name in zip(boxes, names):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # Caja en rojo
            draw.text((x1, y1 - 10), name, fill="red")  # Nombre encima de la caja
            
        # Guardar la imagen en la carpeta
        ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_imagen)
        ruta_imagen = ruta_imagen.replace("\\", "/") 
        imagen.save(ruta_imagen)
        
        # Guardar la imagen en un buffer
        buffer = io.BytesIO()
        imagen.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Codificar la imagen como Base64
        imagen_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Preparar respuesta JSON
        respuesta = {
            "prediccion": [{"nombre": names[i], "puntaje": float(scores[top_indices[i]])} for i in range(len(names))],
            "diagnostico": diagnostico,
            "ruta_imagen": ruta_imagen,
            "imagen": imagen_base64
        }
        
        return jsonify(respuesta)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#--------------- DEVUELVE LA IMAGEN ---------------------------------------------
@app.route('/detectarImagen', methods=['POST'])
def detectarImagen():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400
    
    archivo_imagen = request.files['imagen']
    try:
        # Leer y preprocesar la imagen
        imagen = Image.open(io.BytesIO(archivo_imagen.read())).convert('RGB')
        imagen_preprocesada = preprocesar_imagen(imagen)
        
        # Realizar predicción
        with torch.no_grad():
            predictions = model(imagen_preprocesada)
        
        # Extraer las dos cajas con los puntajes más altos
        scores = []
        boxes = []
        names = []
        for i, box in enumerate(predictions[0]["boxes"]):
            score = predictions[0]["scores"][i].cpu().detach().numpy()
            scores.append(score)
        
        # Ordenar por puntaje y seleccionar los dos más altos
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
        for idx in top_indices:
            box = predictions[0]["boxes"][idx]
            boxes.append(box.cpu().tolist())
            
            label = predictions[0]["labels"][idx].item()
            if label >= len(index2name):
                label = 0  # Etiqueta por defecto si el índice es inválido
            name = index2name[label]
            names.append(name)
        
        # Dibujar las cajas sobre la imagen original
        draw = ImageDraw.Draw(imagen)
        for box, name in zip(boxes, names):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # Caja en rojo
            draw.text((x1, y1 - 10), name, fill="red")  # Nombre encima de la caja
        
        # Guardar la imagen en un buffer para enviarla como respuesta
        buffer = io.BytesIO()
        imagen.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Enviar la imagen con las cajas dibujadas como respuesta
        return send_file(buffer, mimetype='image/png', as_attachment=False, download_name='prediccion.png')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#---------------- DEVUELVE LA PREDICCION -----------------------------------------------------
@app.route('/detectarjson', methods=['POST'])
def detectarjson():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400
    
    archivo_imagen = request.files['imagen']
    try:
        # Leer y preprocesar la imagen
        imagen = Image.open(io.BytesIO(archivo_imagen.read())).convert('RGB')

        imagen_preprocesada = preprocesar_imagen(imagen)
        
        # Realizar predicción
        with torch.no_grad():
            predictions = model(imagen_preprocesada)
        
        # Extraer cajas con el mayor puntaje
        scores = []
        boxes = []
        names = []
        b = False
        for i, box in enumerate(predictions[0]["boxes"]):
            score = predictions[0]["scores"][i].cpu().detach().numpy()
            scores.append(score)
            
            if score == max(scores):  # Seleccionar solo el mayor puntaje
                boxes.append(box.cpu().tolist())  # Convertir a lista
                label = predictions[0]["labels"][i].item()
                if label >= len(index2name):
                    label = 0  # Etiqueta por defecto si el índice es inválido
                name = index2name[label]
                names.append(name)
        
        # Preparar respuesta
        resultados = [
            {"nombre": names[i], "puntaje": float(scores[i]), "caja": boxes[i]}
            for i in range(len(names))
        ]
        
        return jsonify({"prediccion": resultados})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    #app.run(debug=True)
