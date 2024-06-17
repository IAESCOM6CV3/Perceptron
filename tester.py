import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from threading import Semaphore, Thread
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
images = []
labels = []
def ConvertImage(label, path: str, sem: Semaphore):
    global images, labels
    sem.acquire()
    with Image.open(path) as img:
        print(path, label)
        img = img.convert('L')  # Convertir a escala de grises
        img = img.resize((250, 250))  # Redimensionar a 28x28 píxeles
        img_array = np.array(img).flatten()  # Convertir a vector
        images.append(img_array)
        labels.append(label)
    sem.release()
def load_images_from_folder(*args):
    threads = list()
    sem = Semaphore(100)
    counter = 0
    folders = args
    label = 0
    for folder in folders:
        for i in os.listdir(folder):
            subfolder_path = os.path.join(folder, i)
            thread = Thread(target=ConvertImage, args=(label, subfolder_path, sem,))
            threads.append(thread)
            thread.start()
        label += 1
    for thread in threads:
        thread.join()
    return np.array(images), np.array(labels)
# Cargar el modelo guardado
model_filename = 'perceptron_model.joblib'
logistic_regression = joblib.load(model_filename)

X, y = load_images_from_folder(
    'FinalDatasets/AvionesComerciales/images/train'
    #'FinalDatasets/Motos/bike_test'
)
print(len(X), len(y))
# Asegurar que X tenga la forma correcta
X = X.astype(np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Asegurar que y tenga la forma correcta
y = y.astype(np.int32)
# Hacer predicciones
y_pred = logistic_regression.predict(X)
y_pred_proba = logistic_regression.predict_proba(X)

# Evaluar el modelo
accuracy = accuracy_score(y, y_pred)
print(f"Precisión de la regresión logística cargada: {accuracy * 100:.2f}%")

# Ver los valores de salida (probabilidades)
print("Probabilidades de predicción:")
print(y_pred_proba)
#print(max(y_pred_proba.all()))
# Evaluar el modelo
accuracy = accuracy_score(y, y_pred)
print(f"Precisión del perceptrón cargado: {accuracy * 100:.2f}%")
max_probabilities = np.max(y_pred_proba, axis=1)
print("Maximum probabilities for each sample:")
print(max_probabilities)
max_probabilities = [1 if i >= 0.9 else 0 for i in max_probabilities]
count = 0
for i in max_probabilities:
    if i == 1:
        count += 1
percent = count / len(X)
print(max_probabilities, percent)

# Get the class with the highest probability for each sample
predicted_classes = np.argmax(y_pred_proba, axis=1)
#print("Predicted classes for each sample:")
#print(predicted_classes)