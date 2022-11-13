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
