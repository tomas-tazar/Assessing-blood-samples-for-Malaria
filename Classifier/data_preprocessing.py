import os, cv2, numpy as np, keras
from glob import glob

# ========================================================================================================

data = "../Dependencies/malaria/malaria-nih/nih-malaria/"
vgg_data = "../Dependencies/malaria/malaria-nih/vgg/"
parasitized_data = "../Dependencies/malaria/malaria-nih/nih-malaria/parasitized/"
uninfected_data = "../Dependencies/malaria/malaria-nih/nih-malaria/uninfected/"
classes = len(os.listdir(data))

# ========================================================================================================

def preprocess_training_data():
    count, num_of_errors = 0, 0
    num_of_train_samples = 15000
    images, labels = [], []
        
    for file in os.listdir(os.path.join(parasitized_data)):
        if count < num_of_train_samples:
            try:
                img_path = os.path.join(os.path.join(parasitized_data, file))            
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (100, 100)) 
                image = np.asarray(image, dtype="f") / 255.0
                images += [image]
                labels[count] = null_count
                if count % 2500 == 0:
                    print('Processing: {0} images out of {1} in {2}'.format(count, num_of_train_samples, "parasitized"))
                count += 1
            except Exception as e:
                num_of_errors += 1
                continue
            null_count += 1
        else:
            break
    x_images_train = np.array(images)
    #y_labels = np_utils.to_categorical(labels[:num_of_train_samples], 0)
    y_labels = np.array(labels)
    
    return x_images_train, y_labels

# ========================================================================================================

def preprocess_validation_data():
    count, num_of_errors = 0, 0
    num_of_val_samples = 5000
    images, labels = [], []
        
    for file in os.listdir(os.path.join(uninfected_data)):
        if count < num_of_val_samples:
            try:
                img_path = os.path.join(os.path.join(uninfected_data, file))
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (100, 100)) 
                image = np.asarray(image, dtype="f") / 255.0
                images += [image]
                labels[count_val] = null_count_val
                if count_val % 2500 == 0:
                    print('Processing: {0} images out of {1} in {2}'.format(count_val, num_of_val_samples, "uninfected"))
                count_val += 1
            except Exception as e:
                num_of_errors += 1
                continue
            null_count_val += 1
        else:
            break
    x_images_valid = np.array(images)
    #y_labels = np_utils.to_categorical(labels[:num_of_val_samples], 0)
    y_labels = np.array(labels)
    
    return x_images_valid, y_labels

# ========================================================================================================

def process_data():
    print("\n", "=" * 32, "\n", "Loading data for Keras Training", "\n", "=" * 32, "\n")
    x_train_data, y_train_labels = preprocess_training_data()
    x_validation_data, y_validation_labels = preprocess_validation_data()
            
    keras_compatible_pipeline = keras.utils.image_dataset_from_directory(data, image_size=(100,100)) # For custom model
    
    return x_train_data, y_train_labels, x_validation_data, y_validation_labels, keras_compatible_pipeline

# ========================================================================================================

def vgg16_preprocessing():
    print("\n", "=" * 32, "\n", "Loading data for VGG16 Training", "\n", "=" * 32, "\n")
    image_size = [224, 224]
    classes = glob("Dependencies/malaria/malaria-nih/nih-malaria/vgg/*")
    
    return image_size, vgg_data, classes

# ========================================================================================================
