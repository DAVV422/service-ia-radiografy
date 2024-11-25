# SERVICE IA RADIOGRAFY
# Requisitos
- Tener instalado Python

## Instalación

1. Crear un entorno virtual
`    
    python -m venv .venv    
`
2. Ejecutar el entorno virtual
`
    .venv/Scripts/activate
`
3. Instalar las dependencias
`
    pip install -r requirements.txt
`
4. Ejecutar Flask
`
    flask --app serviceIA run
`

## API
Todas las rutas reciben como parametro una imagen 
____
### /detectar
- Devuelve los datos de la prediccion y la ruta de la imagen:
~~~
{
    "prediccion": [
        {"nombre": "Cardiomegaly", "puntaje": 0.95}
    ],
    "ruta_imagen": "./imagenes_procesadas/imagen_procesada.png"
}
~~~
### /detectar2
- Devuelve la imagen en base64 y los datos de la prediccion:
~~~
{
    "prediccion": [
        {"nombre": "Cardiomegaly", "puntaje": 0.95}
    ],
    "imagen": "data:image/png;base64,..."
}
~~~
### /detectarImagen
- Devuelve la imagen con el cuadro de la anomalía dibujado
### /detectarjson
- Devuelve un json con los datos de la predicción:
~~~
{
    "prediccion": [
        {
            "nombre": "nombre_anomalia",
            "puntaje": 0.95,
            "caja": [50.0, 60.0, 200.0, 220.0]
        }
    ]
}
~~~