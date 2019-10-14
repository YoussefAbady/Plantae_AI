# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:38:45 2018

@author: joou
"""
import numpy as np 
import tensorflow as tf
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, concatenate , Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras as k
import matplotlib.pyplot as plt
from PIL import Image


train_data_dir =r'E:\GP\train'
test_data_dir = r'E:\GP\val'
nb_train_samples = 860
nb_validation_samples = 340
nb_epochs = 5
nb_batch_size = 20

# dimensions of our images.
input_shape = np.array([224,224,3])
img_width, img_height = input_shape[0] , input_shape[1]


################################# Model Implementaion #########################

# This is a Stander GoogleNetV1 Inception Layer , 
# Layer Strcture is as follow : 
# Branch_1 : MaxPool_3x3/1(s) -> MaxPoolProjection ( conv_1x1/1(s) with #poolProj_num )
# Branch_2 : conv_1x1/1(s) to Reduce with #conv5x5_red_num -> conv_5x5/1(s) with #conv5x5_num 
# Branch_3: conv_1x1/1(s) to Reduce with #conv3x3_red_num -> conv_3x3/1(s) with #conv3x3_num 
# Branch_4 : conv_1x1/1(s) with #conv1x1_num
def GoogleNEtIncep(inc_name , inc_input , conv1x1_num , conv3x3_red_num , conv3x3_num , conv5x5_red_num , conv5x5_num , poolProj_num):
    
    #Brach_1:
    inc_B1_1 = MaxPooling2D(pool_size=(3,3) ,strides=1,padding='same',name=inc_name+'B1_1')(inc_input)
    inc_B1_2 = Convolution2D(poolProj_num, kernel_size=(1,1),padding='same',activation='relu',name=inc_name+'B1_2')(inc_B1_1)
    
    #Branc_2:
    inc_B2_1 = Convolution2D(conv5x5_red_num, kernel_size=(1,1),padding='same',activation='relu',name=inc_name+'B2_1')(inc_input)
    inc_B2_2 = Convolution2D(conv5x5_num,kernel_size=(5,5),padding='same',activation='relu',name=inc_name+'B2_2')(inc_B2_1)
    
     #Branch_3:
    inc_B3_1 = Convolution2D(conv3x3_red_num,kernel_size=(1,1),padding='same',activation='relu',name=inc_name+'B3_1')(inc_input)
    inc_B3_2 = Convolution2D(conv3x3_num,kernel_size=(1,1),padding='same',activation='relu',name=inc_name+'B3_2')(inc_B3_1)
    
    #Branh_4: 
    inc_B4_1 = Convolution2D(conv1x1_num,kernel_size=(1,1),padding='same',activation='relu',name=inc_name+'B4_1')(inc_input)

    
 
    
    #Concatnating ,, 
    inc_out = concatenate([inc_B4_1,inc_B3_2,inc_B2_2,inc_B1_2],axis=3,name=inc_name+'Out')
    
    return inc_out

def GoogleNetModdel(input_shape , preWightsPath=None ):
    
    x_ip = Input(shape=input_shape)
    
    #Create GoogleNet Model ..
    
    #Layer_1: 
    L1_conv = Convolution2D(64,kernel_size=(7,7),padding='same',strides=2,activation='relu',name='L1_covn')(x_ip)
    L1_Pool = MaxPooling2D(pool_size=(3,3) ,strides=2,padding='same',name='L1_pool')(L1_conv)
    
    #Layer_2: 
    L2_conv_red = Convolution2D(64,kernel_size=(1,1),padding='valid',strides=1,activation='relu',name='L2_conv_red')(L1_Pool)
    L2_conv = Convolution2D(192,kernel_size=(3,3),padding='same',strides=1,activation='relu',name='L2_conv')(L2_conv_red)
    L2_Pool = MaxPooling2D(pool_size=(3,3) ,strides=2,padding='same',name='L2_pool')(L2_conv)
    
    
    #Lyaer_3:Inception_1
    L3_inc = GoogleNEtIncep('L3_inc_1' , L2_Pool , 64 , 96 , 128 , 16 , 32 , 32)
    
    #Layer_4:Inception_2
    L4_inc = GoogleNEtIncep('L4_inc_2' , L3_inc , 128 , 128 , 192 , 32 , 96 , 64)
    L4_Pool = MaxPooling2D(pool_size=(3,3) ,strides=2,padding='same',name='L4_pool')(L4_inc)
    
    #Layer_5:Inception_3
    L5_inc = GoogleNEtIncep('L5_inc_3' , L4_Pool , 192 , 96 , 208 , 16 , 48 , 64)
    
    #Layer_6:Inception_4
    L6_inc = GoogleNEtIncep('L6_inc_4' , L5_inc , 160 , 112 , 224 , 24 , 64 , 64)
    
    #Layer_7:Inception_5
    L7_inc = GoogleNEtIncep('L7_inc_5' , L6_inc , 128 , 128 , 256 , 24 , 64 , 64)
    
    #Layer_8:Inception_6
    L7_inc = GoogleNEtIncep('L7_inc_6' , L7_inc , 112 , 144 , 288 , 32 , 64 , 64)
    
    #Layer_9:Inception_7
    L9_inc = GoogleNEtIncep('L9_inc_7' , L7_inc , 256 , 160 , 320 , 32 , 128 , 128)
    L9_Pool = MaxPooling2D(pool_size=(3,3) ,strides=2,padding='same',name='L9_pool')(L9_inc)
        
    #Layer_10:Inception_8
    L10_inc = GoogleNEtIncep('L10_inc_8' , L9_Pool , 256 , 160 , 320 , 32 , 128 , 128)
    
    #Layer_11:Inception_9
    L11_inc = GoogleNEtIncep('L11_inc_9' , L10_inc , 384 , 192 , 384 , 48 , 128 , 128)
    L11_Pool = AveragePooling2D(pool_size=(7,7) ,strides=1,padding='valid',name='L11_pool')(L11_inc) 
    L11_Flat = Flatten()(L11_Pool)
    L11_Dropout = Dropout(0.4)(L11_Flat)
    classifier_out = Dense(units = 10 ,name='classifier',activation='softmax',kernel_regularizer=l2(0.0002))(L11_Dropout)
    
    model = Model(inputs = x_ip , outputs = classifier_out)
    
    if preWightsPath:
        model.load_weights(preWightsPath)
        
    return model



########################## Data Loader ########################################

def Dataset_loader(train_data_dir ,validation_data_dir):
  
    train_datagen_conf = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    test_datagen_conf = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen_conf.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=nb_batch_size,class_mode='binary')
    test_generator = test_datagen_conf.flow_from_directory(validation_data_dir,target_size=(img_width, img_height),batch_size=nb_batch_size,class_mode='binary')

    return train_generator , test_generator

################################## APP ########################################
    
def classify(imgPath , wieghtsPath) :
    
    
    return res    

################################## Training ###################################

def Train():
    
    train_data , test_data = Dataset_loader(train_data_dir,test_data_dir)
    
    myModel = GoogleNetModdel(input_shape)
    myModel.compile(loss=k.losses.binary_crossentropy , optimizer = k.optimizers.Adadelta() , metrics=['accuracy'] )
    myModel.fit_generator(train_data , epochs = nb_epochs , verbose = 1 , validation_data = test_data )
    
    results = myModel.evaluate(test_data,verbose=0)
    print("Loss : " + results[0])
    print("Test Accuracy : " + results[1])   
    
    return results

###############################################################################` 

#res = classify(r'C:\Users\joou\Google Drive\Plantae_ML_Team\Source\Abady\test_spot.jpg', )


myModel = GoogleNetModdel(input_shape,r'C:\Users\joou\Google Drive\Plantae_ML_Team\Source\Abady\googlenet_10_92.h5')
myModel.compile(loss=k.losses.binary_crossentropy , optimizer = k.optimizers.Adadelta() , metrics=['accuracy'] ) 

img = Image.open(r'C:\Users\joou\Google Drive\Plantae_ML_Team\Source\Abady\test_spot.jpg')
img.load() 
img = img.resize((224,224))
img_mat = np.asarray(img,dtype='uint8')
img_mat = img_mat.reshape(1,256,256,3)

res = myModel.predict(img_mat); 




















