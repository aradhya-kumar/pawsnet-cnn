import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/pawsnet_model.h5")

img = cv2.imread("sample.jpg")
img = cv2.resize(img, (256,256))
img = img / 255.0
img = np.reshape(img, (1,256,256,3))

prediction = model.predict(img)[0][0]

if prediction > 0.5:
    print("The Given Image is of a Cat ")
else:
    print("The Given Image is of a Dog")
