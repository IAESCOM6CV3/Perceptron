from PIL import Image
import os
imSize = 150
carTrainImages = os.listdir("Datasets/Coches/cars_train/cars_train")
carTestImages = os.listdir("Datasets/Coches/cars_test/cars_test")
bikeImages = os.listdir("Datasets/CarVSMoto/Car-Bike-Dataset/Bike")
file = open("Datasets/AvionesComerciales/train.csv","r").readlines()
comercialPlaneDataset = [f"{line.strip().split(',')[0]}" for line in file]
file = open("Datasets/AvionesComerciales/test.csv","r").readlines()
comercialPlaneTestDataset = [f"{line.strip().split(',')[0]}" for line in file]
if not os.path.exists("FinalDatasets/"):
    os.mkdir("FinalDatasets/")
if not os.path.exists("FinalDatasets/AvionesComerciales"):
    os.mkdir("FinalDatasets/AvionesComerciales")
    os.mkdir("FinalDatasets/AvionesComerciales/images")
    os.mkdir("FinalDatasets/AvionesComerciales/images/test")
    os.mkdir("FinalDatasets/AvionesComerciales/images/train")
if not os.path.exists("FinalDatasets/Coches"):
    os.mkdir("FinalDatasets/Coches")
    os.mkdir("FinalDatasets/Coches/cars_train")
    os.mkdir("FinalDatasets/Coches/cars_test")
if not os.path.exists("FinalDatasets/Motos"):
    os.mkdir("FinalDatasets/Motos")
    os.mkdir("FinalDatasets/Motos/bike_test")
    os.mkdir("FinalDatasets/Motos/bike_train")
def ResizeImage(pathOrigin, pathDestiny, image):
    try:
        photo = Image.open(f"{pathOrigin}{image}")
        newPhoto = photo.resize([imSize,imSize])
        img = newPhoto.save(f"{pathDestiny}{image}")
        print("Image resized", image)
    except Exception as ex:
        print(str(ex))
from threading import Thread
count = 0
import random
limit = 1500
datasetFile = open("FinalDatasets/Coches/train.csv",'a')
datasetFile.write("Imagen,Ruta,Ruedas,Puertas,Ventanas,Color\n")
for i in carTrainImages:
    if limit > count:
        random_int = random.randint(0, 1)
        random_int2 = random.randint(2, 4)
        random_int3 = random.randint(1, 3)
        colores = ['Blanco', 'Rojo', 'Verde', 'Azul', 'Negro']
        datasetFile.write(f"{i},FinalDatasets/Coches/cars_train/,4,{2 if random_int2 == 0 else 4},{2 if random_int3 == 1 else 4},{colores[random.randint(0, 4)]}\n")
        ResizeImage("Datasets/Coches/cars_train/cars_train/", "FinalDatasets/Coches/cars_train/", i)
        count += 1
    else:
        break
count = 0
datasetFile.close()
datasetFile = open("FinalDatasets/Coches/test.csv",'a')
datasetFile.write("Imagen,Ruta,Ruedas,Puertas,Ventanas,Color\n")
for i in carTestImages:
    if limit > count:
        random_int = random.randint(0, 1)
        random_int2 = random.randint(2, 4)
        random_int3 = random.randint(1, 3)
        colores = ['Blanco', 'Rojo', 'Verde', 'Azul', 'Negro']
        datasetFile.write(f"{i},FinalDatasets/Coches/cars_test/,4,{2 if random_int2 == 0 else 4},{2 if random_int3 == 1 else 4},{colores[random.randint(0, 4)]}\n")
        ResizeImage("Datasets/Coches/cars_test/cars_test/", "FinalDatasets/Coches/cars_test/", i)
        count+= 1
    else:
        break
count = 0
datasetFile.close()
datasetFile = open("FinalDatasets/AvionesComerciales/train.csv",'a')
datasetFile.write("Imagen,Ruta,Alas,Puertas,Motores,Color\n")
for i in comercialPlaneDataset:
    if limit > count:
        random_int = random.randint(0, 1)
        random_int2 = random.randint(1, 4)
        random_int3 = random.randint(1, 3)
        colores = ['Blanco', 'Rojo', 'Gris', 'Perla']
        datasetFile.write(f"{i},FinalDatasets/AvionesComerciales/images/train/,{2 if random_int == 1 else 4},{random_int2},{random_int3},{colores[random.randint(0, 3)]}\n")
        ResizeImage("Datasets/AvionesComerciales/images/","FinalDatasets/AvionesComerciales/images/train/", i)
        count += 1
    else: 
        break
count = 0
datasetFile.close()
datasetFile = open("FinalDatasets/AvionesComerciales/test.csv",'a')
datasetFile.write("Imagen,Ruta,Alas,Puertas,Motores,Color\n")
for i in comercialPlaneTestDataset:
    if limit > count:
        random_int = random.randint(0, 1)
        random_int2 = random.randint(1, 4)
        random_int3 = random.randint(1, 3)
        colores = ['Blanco', 'Rojo', 'Gris', 'Perla']
        datasetFile.write(f"{i},FinalDatasets/AvionesComerciales/images/test/,{2 if random_int == 1 else 4},{random_int2},{random_int3},{colores[random.randint(0, 3)]}\n")
        ResizeImage("Datasets/AvionesComerciales/images/","FinalDatasets/AvionesComerciales/images/test/", i)
        count += 1
    else:
        break
count = 0
datasetFile.close()
datasetFile = open("FinalDatasets/Motos/train.csv",'a')
datasetFile.write("Imagen,Ruta,Ruedas,Color,Cilindraje\n")
datasetFile1 = open("FinalDatasets/Motos/test.csv",'a')
datasetFile1.write("Imagen,Ruta,Ruedas,Color,Cilindraje\n")
for i in bikeImages:
    if (limit * 2) > count :
        random_int = random.randint(0, 1)
        random_int2 = random.randint(2, 4)
        random_int3 = random.randint(1, 3)
        colores = ['Blanco', 'Rojo', 'Verde', 'Azul', 'Negro']
        if count % 2 == 0:
            ResizeImage("Datasets/CarVSMoto/Car-Bike-Dataset/Bike/","FinalDatasets/Motos/bike_test/", i)
            datasetFile1.write(f"{i},FinalDatasets/Motos/bike_test/,2,{colores[random.randint(0, 3)]},{125 if random_int == 0 else 250}\n")
        else:
            ResizeImage("Datasets/CarVSMoto/Car-Bike-Dataset/Bike/","FinalDatasets/Motos/bike_train/", i)
            datasetFile.write(f"{i},FinalDatasets/Motos/bike_train/,2,{colores[random.randint(0, 3)]},{125 if random_int == 0 else 250}\n")
        
    count += 1