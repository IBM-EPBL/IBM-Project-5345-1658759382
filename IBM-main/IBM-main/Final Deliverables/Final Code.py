

# In[1]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(
     r'C:\Users\user\Downloads\dataset\TRAIN_SET\TRAIN_SET',
     target_size=(64,64),batch_size=5,color_mode='rgb',class_mode='sparse')
x_test=test_datagen.flow_from_directory(
     r'C:\Users\user\Downloads\dataset\TEST_SET\TEST_SET',
     target_size=(64,64),batch_size=5,color_mode='rgb',class_mode='sparse')


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


# In[3]:


img=image.load_img(r"C:\Users\user\Downloads\dataset\TRAIN_SET\TRAIN_SET\ORANGE\0_100.jpg",
grayscale=False,target_size=(64, 64)) #loading ef the mage
x=image.img_to_array(img) # image to array
x=np.expand_dims(x,axis = 0)#chang ing the shape
pred=model.predict(x) #predicting the cLasses
pred


# In[ ]:


# Flask-It is our framework which we are going to use to run/serve our app
#request-for accessing file which was uploaded by the user on our applicat
import os

import numpy as np  # used for numerical analysis
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model  # to load our trained model
from tensorflow.keras.preprocessing import image

app= Flask (__name__,template_folder= "templates") # initializing a flask apPp
#Loading the model
model=load_model('nutrition.h5')
print("Loaded model from disk")

@app.route("/")# route to display the home page
def home():
    return render_template("home.html")#rendering the home page
@app.route('/image 1 ',methods=['GET','POST'])# routes to the index html
def image1():
    return render_template("image.htmL")

@app.route('/predict',methods=['GET','POST'])# route to show the predictions in a
def launch():
    if request.method=='POST':
       print("Hi")
       f=request.files['image'] #requesting the file
       print("hi")
       basepath=os.path.dirname ('__file__')#storing the file directory
       filepath=os.path.join(basepath,"uploads",f.filename)#storing the file in uploads folder
       print(filepath)
       f.save(filepath) #s aving the file

       img=image.load_img(filepath, target_size=(64, 64)) #load and reshaping the image
       x=image.img_to_array(img)#converting image to an array
       x=np.expand_dims(x,axis=0) #changing the dimensions of the image
  
       pred=np.argmax(model.predict(x), axis=1)
       print("prediction",pred)#printing the prediction
       index=['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE ', 'WATERMELON']
     
       result=str(index[pred[0]])
      
       X=result
       print(x)
      
       result=nutrition(result)
       print(result)

       return render_template("imageprediction.html", showcase=(result))

def nutrition (index):
    
    url="https://calorieninjas.p.rapidapi.com/v1/nutrition"
    
    querystring={"query": index}
    
    headers = {
    'x-rapidapi-key': "5d797ab107mshe668f26bd044e64p1ffd34jsnf47bfa9a8ee4",
    'x-rapidapi-host': "calorieninjas.p.rapidapi.com"  
    }
    response=requests.request("GET",url, headers=headers, params=querystring)
    
    print(response.text)
    return response.json()["items"]
   
if __name__ == "__main__":
    #running the app
    app.run(debug=False)


# In[ ]:




