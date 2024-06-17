from flask import Flask, render_template, request
import os
import random
import base64
import numpy as np
from io import BytesIO
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from threading import Semaphore, Thread
import numpy as np
from sklearn.linear_model import LogisticRegression
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
app = Flask(__name__)
def ConvertImage(images, labels):
    imagesArray = list()
    labelsArray = list()
    for image, label in zip(images, labels):
        try:
            image = str(image).strip()
            decodedImage = DecodeImage(image)
            image = BytesIO(decodedImage)
            print("Here")
            with Image.open(image) as img:
                img = img.convert('L')  # Convertir a escala de grises
                img = img.resize((250, 250))  # Redimensionar a 28x28 píxeles
                img_array = np.array(img).flatten()  # Convertir a vector
                imagesArray.append(img_array)
                labelsArray.append(label)
        except Exception as ex:
            print(str(ex), image, label)
    print("Return")
    return np.array(imagesArray), np.array(labelsArray)
def LoadModels():
    bikeCarModel = 'bike_car.joblib'
    bikeCar = joblib.load(bikeCarModel)
    planeCarModel = 'plane_car.joblib'
    planeCar = joblib.load(planeCarModel)
    return bikeCar, planeCar
def EvaluateImages(X, y, model):
    print(len(X), len(y))
    X = X.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Asegurar que y tenga la forma correcta
    y = y.astype(np.int32)
    # Hacer predicciones
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    accuracy = accuracy_score(y, y_pred)
    accuracy = accuracy * 100
    print(f"Precisión del perceptrón cargado: {accuracy:.2f}%")
    max_probabilities = np.max(y_pred_proba, axis=1)
    max_probabilities = [1 if i >= 0.99 else 0 for i in max_probabilities]
    count = 0
    for i in max_probabilities:
        if i == 1:
            count += 1
    percent = count / len(X)
    return accuracy, percent, max_probabilities
@app.route("/api/Evaluate/Images", methods=['POST'])
def EvaluateImagesFunc():
    try:
        bikeCarModel, planeCarModel = LoadModels()
        data = request.get_json()
        images = data['images']
        labels = data['labels']
        images = [str(image) for image in images]
        X, y = ConvertImage(images, labels)
        print("Images loaded")
        accuracyScoreBikes, percentScoreBikes, probabilitiesBikes = EvaluateImages(X, y, bikeCarModel)
        accuracyScorePlanes, percentScorePlanes, probabilitiesPlanes = EvaluateImages(X, y, planeCarModel)
        return {"Accuracy": {"Bikes": accuracyScoreBikes, "Planes": accuracyScorePlanes, "Cars": (accuracyScoreBikes + accuracyScorePlanes)/2},
                "RealPercent": {"Bikes": percentScoreBikes, "Planes": percentScorePlanes, "Cars": (percentScoreBikes + percentScorePlanes)/2},
                "Probabilities": {"Bikes": probabilitiesBikes, "Planes": probabilitiesPlanes, "Cars": probabilitiesPlanes}}, 200
    except Exception as ex:
        return {"Message": str(ex)}, 500

@app.route('/')
def home():
    return render_template('index.html')
def EncodeImage(imagePath: str):
    with open(imagePath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string
def DecodeImage(image: str):
    return base64.b64decode(image)
@app.route("/api/Get/Images/Train", methods=['GET'])
def GetImagesFromDatasets():
    try:
        planes = os.listdir("FinalDatasets/AvionesComerciales/images/train")
        cars = os.listdir("FinalDatasets/Coches/cars_train")
        bikes = os.listdir("FinalDatasets/Motos/bike_train")
        planesReturn = [EncodeImage(f"FinalDatasets/AvionesComerciales/images/train/{planes[random.randint(0, len(planes)-1)]}") for i in range(0, 5)]
        carsReturn = [EncodeImage(f"FinalDatasets/Coches/cars_train/{cars[random.randint(0, len(cars)-1)]}") for i in range(0, 5)]
        bikesReturn = [EncodeImage(f"FinalDatasets/Motos/bike_train/{bikes[random.randint(0, len(bikes)-1)]}") for i in range(0, 5)]
        return {"planes": planesReturn, "cars": carsReturn, "bikes": bikesReturn}, 200
    except Exception as ex:
        return {"Error": str(ex)}, 500
@app.route("/api/Test/Image", methods=['GET'])
def TestImage():
    try:
        image = request.args.get('image')
        category = request.args.get('category')
        return {"image": image, "category": category}, 200
    except Exception as ex:
        return {"Error": str(ex)}, 500
if __name__ == '__main__':
    app.run(debug=True)
