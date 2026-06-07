import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

def arsitekturCNN():
    input1 = keras.Input(shape=(10, 12, 1))
    input2 = keras.Input(shape=(46, 12, 1))

    x = layers.Conv2D(filters=16,kernel_size=(2,12))(input1)  
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32,kernel_size=(2,1))(x)    
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(x) 
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1,outputs=x)

    y = layers.Conv2D(filters=16,kernel_size=(15,12))(input2)    
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  
    y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  
    y = layers.Flatten()(y)
    y = keras.Model(inputs=input2,outputs=y)

    combined = layers.concatenate([x.output,y.output])
    z = layers.Dense(128,activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1,activation='sigmoid')(z)        

    model = keras.Model(inputs=[input1,input2],outputs=z)
    return model
