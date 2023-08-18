import tensorflow as tf, numpy as np
from data_preprocessing import vgg16_preprocessing
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, Input, Lambda, Reshape, Multiply
from keras.models import Model
from keras.models import Sequential
from keras.applications.vgg16 import VGG16

# ========================================================================================================

def initialize_custom_model():    
    model = Sequential()

    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    # tf.keras.utils.plot_model(model, show_shapes=True, to_file='Keras_Sequential_model') #to visualize the model
    return model

# ========================================================================================================

def initialize_sequential_v2_model():
    def spatial_attention(input_feature_map):
        # Spatial Attention mechanism function to focus on specific parts of the image
        # Global Average Pooling
        avg_pool = GlobalAveragePooling2D()(input_feature_map)
        # Reshape to (batch_size, channels)
        reshaped = Reshape((1, 1, input_feature_map.shape[3]))(avg_pool)
        # Convolution with 1x1 filters and 'sigmoid' activation
        conv = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(reshaped)
        # Element-wise multiplication between input feature map and attention map
        multiplied = Multiply()([input_feature_map, conv])
        return multiplied

    inputs = Input(shape=(100, 100, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)

    # Spatial Attention Mechanism
    attention = spatial_attention(conv3)
    dense = Dense(256, activation='relu')(Flatten()(attention))
    dropout = Dropout(0.5)(dense)
    dense2 = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=inputs, outputs=dense2)
    
    return model

# ========================================================================================================

def initialize_vgg16_model():
    image_size, _, classes = vgg16_preprocessing()
    
    vgg = VGG16(input_shape=image_size + [3], weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
        
    x = Flatten()(vgg.output)
    prediction = Dense(len(classes), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # tf.keras.utils.plot_model(model, show_shapes=True, to_file='VGG16_model') #to visualize the model
    return model

# ========================================================================================================
  
