import tensorflow as tf
import cv2
from tensorflow import keras
from matplotlib import pyplot as plt
import os
import numpy as np

def gen_pred(test_dir, img, model):
    rgb_path = os.path.join(base_directory + "/data/test/", img)
    img = cv2.imread(rgb_path)
    img = tf.expand_dims(img, axis=0)
    img = resize(img)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred

def resize(input_image):
   input_image = tf.image.resize(input_image, (256, 256), method="nearest")
   return input_image

base_directory = ""
test_directory = os.listdir(base_directory + "/data/test/")
test_imgs = ['00dc34840.jpg', '00c3db267.jpg', '00aa79c47.jpg', '00a3a9d72.jpg']
fullres_model = keras.models.load_model('trained_models/trained_model.h5', compile=False)

rows = 1
columns = 2
for i in range(len(test_imgs)):
    img, pred = gen_pred(test_directory, test_imgs[i], fullres_model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction")
    plt.show()