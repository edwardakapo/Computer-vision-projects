#Author Oluwademilade Edward Akapo
#Student No: 10195403
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import enum
import numpy as np
import tensorflow as tf
import keras
from keras import datasets, layers, models
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import sklearn
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score

# the names for the labels of our images
str_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load up the data 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the datasets to be between 0 and 1
x_train , x_test = x_train / 255.0 , x_test / 255.0

# show the figure
# plt.figure(figsize = (16,16))
# for i in range(100):
#   plt.subplot(10,10,1+i)
#   plt.axis('off')
#   plt.imshow(x_train[i], cmap = 'gray')
# plt.show()

# training Validating and splitting data to use for data augementation training
# creating validation set with 10,000 images from our training set
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

# create real time data augumentation 
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True
)
# train_datagen.fit(x_train)

test_datagen = ImageDataGenerator()
# val_datagen.fit(x_val)

batch_size = 8
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
valid_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(x_test, y_test, batch_size=1)


# build the CNN model
model = models.Sequential()
#first stack
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
#second stack
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
# final stack
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10))
model.summary()

# Compile and train

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 1st training on the model with no augumentation
model.fit(x_train, y_train, epochs=15, 
                    validation_data=(x_val, y_val))

# 2nd training with intervals of augumented data 
history = model.fit_generator(train_generator, validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // batch_size , epochs=5)

# 3nd training with no augumentation
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_val, y_val))
# Evaluate model 
score = model.evaluate_generator(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.figure()
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.plot(acc,color = 'purple',label = 'Training Acuracy')
# plt.plot(val_acc,color = 'blue',label = 'Validation Accuracy')
# plt.legend()

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure()
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.plot(loss,color = 'green',label = 'Training Loss')
# plt.plot(val_loss,color = 'red',label = 'Validation Loss')
# plt.legend()

#save model to be used by GUI
model.save('image_classifier.h5')
