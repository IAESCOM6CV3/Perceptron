import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from threading import Thread, Semaphore
from sklearn.linear_model import LogisticRegression
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
# Ruta a tu conjunto de datos
print("Loading datasets...")
# Cargar las imágenes de entrenamiento
X, y = load_images_from_folder(
    'FinalDatasets/AvionesComerciales/images/train',
    'FinalDatasets/Coches/cars_train',
)

# Asegurar que X tenga la forma correcta
X = X.astype(np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X)
images = []
labels = []
X_test, y_test = load_images_from_folder(
    'FinalDatasets/AvionesComerciales/images/test',
    'FinalDatasets/Coches/cars_test'
)
X_test = X_test.astype(np.float64)
X_test = scaler.transform(X_test)
# Asegurar que y tenga la forma correcta
y = y.astype(np.int32)
y_test = y.astype(np.int32)
X_train = X
y_train = y

# Crear el modelo de Perceptrón (Con regresión logística)
# Se usa regresión ya que permite la simplificación de características ya que cumple el mismo objetivo que el Perceptrón
# Pero requiriendo de menos información para los pesos
# En el perceptron es Sum(w * x + b > 0)
# En regresión logística es 1 / (1 + e[Sum(W*X)])
logistic_regression = LogisticRegression(max_iter=10000, random_state=10, verbose = 1)

# Entrenar el modelo
logistic_regression.fit(X_train, y_train)

# Hacer predicciones
y_pred = logistic_regression.predict(X_test)
y_pred_proba = logistic_regression.predict_proba(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del perceptrón: {accuracy * 100:.2f}%")

# Guardar el modelo en un archivo
model_filename = 'plane_car.joblib'
#joblib.dump(perceptron, model_filename)
joblib.dump(logistic_regression, model_filename)