from joblib import load
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
img_height = 32
img_width = 55
model_from_joblib = load("my_model")
path = "./Data/animals/images/cats_00060.jpg"
img = load_img(path, target_size=(img_height, img_width))

img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model_from_joblib.predict(img_array)
print(predictions)
