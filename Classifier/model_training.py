import tensorflow as tf, time, os, numpy as np
from keras.preprocessing.image import ImageDataGenerator
from custom_model import initialize_custom_model, initialize_sequential_v2_model, initialize_vgg16_model
from data_preprocessing import process_data, vgg16_preprocessing

# ========================================================================================================

def sequential_training_model():
    time_start = time.time()
    epochs = 20
    batchesNUM = 32

    callback = tf.keras.callbacks.TensorBoard(log_dir="../Dependencies/model_information/model_logs/")

    sequential_model = initialize_custom_model()
    
    train_data, _, validation_data, _, keras_data = process_data()
    
    # train_shuffle = tf.random.shuffle(train_data, seed=22)
    # validation_shuffle = tf.random.shuffle(validation_data, seed=22)
    # train_labels_shuffle = tf.random.shuffle(train_labels, seed=22)
    # validation_labels_shuffle = tf.random.shuffle(validation_labels, seed=22)
    
    keras_data = keras_data.map(lambda x, y: (x / 255.0, y))

    train_data = keras_data.take(500)
    validation_data = keras_data.skip(500).take(262)
    testing_data = keras_data.skip(750).take(100)
    
    print("=" * 20)
    print("Model is training...")
    print("=" * 20)

    model_history = sequential_model.fit(train_data, epochs=epochs, batch_size=batchesNUM, validation_data=(validation_data), callbacks=[callback])
    # model.save(os.path.join('model_information/models/', 'model_Keras_sequential.h5')) # Uncomment to save the model
    print('Training time: %s\n' % (time.time()-time_start))
    return sequential_model, model_history, epochs, batchesNUM, testing_data

# ========================================================================================================

def sequential_training_model_v2():
    time_start = time.time()    
    epochs = 20
    batchesNUM = 32
    
    callback = tf.keras.callbacks.TensorBoard(log_dir="../Dependencies/model_information/model_logs/")

    sequential_model_spatialattention = initialize_sequential_v2_model()
    train_data, _, validation_data, _, keras_data = process_data()
    keras_data = keras_data.map(lambda x, y: (x / 255.0, y))
    train_data = keras_data.take(500)
    validation_data = keras_data.skip(500).take(262)
    testing_data = keras_data.skip(750).take(100)
    
    print("\n", "=" * 32, "\n", "Sequential model is training...", "\n", "=" * 32, "\n")

    model_history = sequential_model_spatialattention.fit(train_data, epochs=epochs, batch_size=batchesNUM, validation_data=(validation_data), callbacks=[callback])
    # model.save(os.path.join('model_information/models/', 'model_Keras_sequential_v2')) # Uncomment to save the model
    print('Training time: %s\n' % (time.time()-time_start))
    return sequential_model_spatialattention, model_history, epochs, batchesNUM, testing_data

# ========================================================================================================
   
def train_vgg16():
    time_start = time.time()
    epochs = 10
    batchesNUM = 32
    
    callback = tf.keras.callbacks.TensorBoard(log_dir="../Dependencies/model_information/model_logs/")
    vgg_model = initialize_vgg16_model()
    _, vgg_data, _ = vgg16_preprocessing()
    
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory(vgg_data, target_size = (224, 224), batch_size = batchesNUM, class_mode='categorical')
    test_set = test_datagen.flow_from_directory(vgg_data, target_size = (224, 224), batch_size = batchesNUM, class_mode='categorical')

    print("\n", "=" * 32, "\n", "VGG16 model is training...", "\n", "=" * 32, "\n")


    trained_model = vgg_model.fit(training_set, validation_data=test_set, epochs=epochs, steps_per_epoch=len(training_set), validation_steps=len(test_set), callbacks=[callback])
    # vgg_model.save(os.path.join('../model_information/models/', 'VGG16_model.h5')) # Uncomment to save the model
    print('Training time: %s\n' % (time.time()-time_start))
    return vgg_model ,trained_model, epochs, batchesNUM, test_set

# ========================================================================================================
