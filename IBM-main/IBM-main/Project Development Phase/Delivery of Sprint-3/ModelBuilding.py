# In[2]:


import numpy as np  # used for numerical analysts
import tensorflow  # open source used for both ML ond DL for computot ion
#MaxPooling20-for downsampling the image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import \
    layers  # 4 Layer consists of a tensor-in tensor-out computation function
#Faltten-used fot flottening the input or change the dimension
#Dense Loyer is the regular deepLy connected neural ne twork Layer
from tensorflow.keras.layers import (Conv2D, Dense,  # Convolutiona t Loyer
                                     Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential  # it is a plain stack of Layers

model=Sequential()
#Initializing the CNN
classifier=Sequential ()
# First convolution Layer and pool ing
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation= 'relu' ))
classifier.add(MaxPooling2D(pool_size= (2, 2)))
#Second convolution Layer and pool ing
classifier.add(Conv2D(32, (3, 3), activation='relu'))
#input shape is going to be the pooled feature maps from the previous convolution Layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening the layers
classifier.add(Flatten())
classifier.add(Dense (units=128, activation='relu' ))
classifier.add(Dense (units=5, activation='softmax'))
classifier.summary()
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
classifier.fit_generator(
        generator=x_train,steps_per_epoch = len(x_train),
        epochs=1, validation_data=x_test, validation_steps=len(x_test)) # No of tmaes tn test set
classifier.save('nutrition.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model= load_model("nutrition.h5") #Loading the modet for testing

img=image.load_img(r"C:\Users\user\Downloads\dataset\TRAIN_SET\TRAIN_SET\ORANGE\0_100.jpg",
grayscale=False,target_size=(64, 64)) #loading ef the mage
x=image.img_to_array(img) # image to array
x=np.expand_dims(x,axis = 0)#chang ing the shape
pred=model.predict(x) #predicting the cLasses
pred
