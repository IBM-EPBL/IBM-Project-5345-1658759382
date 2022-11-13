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
