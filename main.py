import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from threading import Thread, Semaphore
images = []
labels = []
def ConvertImage(label, path: str, sem: Semaphore):
    global images, labels
    sem.acquire()
    with Image.open(path) as img:
        print(path, label)
        img = img.convert('L')  # Convertir a escala de grises
        img = img.resize((125, 125))  # Redimensionar a 28x28 píxeles
        img_array = np.array(img).flatten()  # Convertir a vector
        images.append(img_array)
        labels.append(label)
    sem.release()
def load_images_from_folder(folder, folder2 = None):
    threads = list()
    sem = Semaphore(100)
    counter = 0
    dirs = os.listdir(folder)
    for i in os.listdir(folder2):
        dirs.append(i)
    for label, subfolder in enumerate(os.listdir(folder)):
        
        subfolder_path = os.path.join(folder, subfolder)
        thread = Thread(target=ConvertImage, args=(label, subfolder_path, sem,))
        threads.append(thread)
        thread.start()
        counter += 1
    for thread in threads:
        thread.join()
    return np.array(images), np.array(labels)
# Ruta a tu conjunto de datos
data_folder = 'FinalDatasets/AvionesComerciales/images'
print("Loading datasets...")
# Cargar las imágenes
X, y = load_images_from_folder(data_folder)
print(len(X), len(y))
# Asegurar que X tenga la forma correcta
X = X.astype(np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Asegurar que y tenga la forma correcta
y = y.astype(np.int32)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=1000)

# Crear el modelo de Perceptrón
perceptron = Perceptron(max_iter=400, eta0=1.0, random_state=1000)

# Entrenar el modelo
perceptron.fit(X_train, y_train)

# Hacer predicciones
y_pred = perceptron.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del perceptrón: {accuracy * 100:.2f}%")
