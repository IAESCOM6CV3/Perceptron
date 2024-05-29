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
for i in carTrainImages:
    ResizeImage("Datasets/Coches/cars_train/cars_train/", "FinalDatasets/Coches/cars_train/", i)
for i in carTestImages:
    ResizeImage("Datasets/Coches/cars_test/cars_test/", "FinalDatasets/Coches/cars_test/", i)
for i in comercialPlaneDataset:
    ResizeImage("Datasets/AvionesComerciales/images/","FinalDatasets/AvionesComerciales/images/train/", i)
for i in comercialPlaneTestDataset:
    ResizeImage("Datasets/AvionesComerciales/images/","FinalDatasets/AvionesComerciales/images/test/", i)
count = 0
for i in bikeImages:
    if count % 2 == 0:
        ResizeImage("Datasets/CarVSMoto/Car-Bike-Dataset/Bike/","FinalDatasets/Motos/bike_test/", i)
    else:
        ResizeImage("Datasets/CarVSMoto/Car-Bike-Dataset/Bike/","FinalDatasets/Motos/bike_train/", i)
    count += 1