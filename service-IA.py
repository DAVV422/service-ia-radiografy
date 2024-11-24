from flask import Flask, request, jsonify, send_file, send_from_directory
from torchvision import transforms
from PIL import Image, ImageDraw
from joblib import load
import numpy as np
import base64
import torch
import io
import os

app = Flask(__name__)

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
    # Añadir más etiquetas según tu modelo
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


#---------------- Devuelve un JSON con la imagen y la prediccion -------------------
@app.route('/detectar2', methods=['POST'])
def detectar2():
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
        
        # Guardar la imagen en un buffer
        buffer = io.BytesIO()
        imagen.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Codificar la imagen como Base64
        imagen_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Preparar respuesta JSON
        respuesta = {
            "prediccion": [{"nombre": names[i], "puntaje": float(scores[top_indices[i]])} for i in range(len(names))],
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
    

#----------------- Clasificacion de la imagen, solo devuelve la posicion de la clasificacion -------------------
# Preprocesa la imagen para cumplir con los requisitos del modelo
def preprocesar_imagen2(imagen):
    # Redimensiona a 320x320
    imagen = imagen.resize((320, 320))
    # Convertir a escala de grises
    imagen = imagen.convert('L')
    # Convertir a un array de NumPy y normalizar
    imagen_array = np.array(imagen) / 255.0
    # Redimensionar a (1, 320, 320, 1) para cumplir con la entrada del modelo
    imagen_array = np.expand_dims(imagen_array, axis=(0, -1))
    return imagen_array

# Carga el modelo
modelo = load('./modelo_clasificacion.joblib')

@app.route('/clasificar', methods=['POST'])
def clasificar_imagen():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400
    
    archivo_imagen = request.files['imagen']
    imagen = Image.open(io.BytesIO(archivo_imagen.read()))
    
    # Preprocesar la imagen
    imagen_preprocesada = preprocesar_imagen2(imagen)
    
    # Clasificar la imagen con el modelo cargado
    prediccion = modelo.predict(imagen_preprocesada)
    
    # Retornar la predicción como JSON
    return jsonify({"prediccion": int(np.argmax(prediccion))})
#-----------------------------------

if __name__ == '__main__':
    app.run(debug=True)
