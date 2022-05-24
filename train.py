
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from joblib import dump

path = "./Data/animals/animals/"
batch_size= 16
img_height = 32
img_width = 55

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=True)

baseline_model = Sequential()

baseline_model.add(Convolution2D(32, (3, 3), activation='relu', padding="same", input_shape=(img_height, img_width, 3)))
baseline_model.add((MaxPooling2D(pool_size=(2, 2))))
baseline_model.add(Convolution2D(32, (3, 3), activation='relu', padding="same"))
baseline_model.add((MaxPooling2D(pool_size=(2, 2))))
baseline_model.add(Flatten())
baseline_model.add(Dense(128, activation='relu'))
baseline_model.add(Dense(3, activation='softmax'))


baseline_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

baseline_model.fit(train_generator, epochs=9, verbose=1, batch_size=batch_size)

dump(baseline_model, 'my_model')