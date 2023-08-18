import cv2, numpy as np, tensorflow as tf, os, sys

# ================================================================================================================= #

def evaluated_image(array_of_rois):
    trained_model = tf.keras.models.load_model('Dependencies/model_information/models/model_Keras_sequential.h5')
    sys.stdout = open(os.devnull, 'w')
    predictions = []
    
    for roi in array_of_rois: 
        roi_resized = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
        prediction = trained_model.predict(np.array([roi_resized]))[0]
        predictions.append(prediction)
    
    sys.stdout = sys.__stdout__
    return predictions

# ================================================================================================================= #

def evaluate_image_for_sequential_spatialattention(array_of_rois):
    keras_spatialattention_model = tf.keras.models.load_model('Dependencies/model_information/models/model_Keras_sequential_v2.h5')
    sys.stdout = open(os.devnull, 'w')
    predictions_improved = []
    
    for roi in array_of_rois: 
        roi_resized = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
        improved_pred = keras_spatialattention_model.predict(np.array([roi_resized]))[0]
        predictions_improved.append(improved_pred)
    
    sys.stdout = sys.__stdout__
    return predictions_improved

# ================================================================================================================= #

def evaluate_image_for_vgg(array_of_rois):
    vgg_model = tf.keras.models.load_model('Dependencies/model_information/models/VGG16_model.h5')
    sys.stdout = open(os.devnull, 'w')
    vgg_predictions = []
    
    for roi in array_of_rois: 
        vgg_roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
        vgg_prediction = vgg_model.predict(np.array([vgg_roi_resized]))[0]
        vgg_predictions.append(vgg_prediction)
    
    sys.stdout = sys.__stdout__
    return vgg_predictions

# ================================================================================================================= #

def display_labelled_image(contours, predictions, image_with_rois):
    position_inf = (20, 25) # width, height
    position_inf_num = (95, 25)
    position_uninf = (20, 50)
    position_uninf_num = (110, 50)
    position_unk = (20, 75)
    position_unk_num = (100, 75)
    box_position = (10, 10)
    box_position_end = (145, 85) 
    
    number_of_infected = 0
    number_of_uninfected = 0
    unk = 0
    total = 0
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        total += 1
        if predictions[i] == 0:
            colour = (0, 0, 255) # Red = infected
            number_of_infected += 1
        elif predictions[i] == 1:
            colour = (0, 255, 0) # Green = uninfected
            number_of_uninfected += 1
        elif predictions[i] <= 0.5:
            colour = (0, 0, 255) # Red = infected (probabilistic)
        else:
            colour = (255, 0, 0) # Blue = uninfected
            unk += 1

        cv2.rectangle(image_with_rois, (x, y), (x + w, y + h), colour, 2)

    cv2.rectangle(image_with_rois, box_position, box_position_end, (50,50,50), -1)
    cv2.putText(image_with_rois, "Infected:", position_inf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image_with_rois, str(number_of_infected), position_inf_num, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image_with_rois, "Uninfected:", position_uninf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image_with_rois, str(number_of_uninfected), position_uninf_num, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image_with_rois, "Unknown:", position_unk, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(image_with_rois, str(unk), position_unk_num, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return image_with_rois, number_of_infected, number_of_uninfected, unk, total

# ================================================================================================================= #

def vgg_display_labelled_image(contours, predictions, image_with_rois):
    position_inf = (20, 25) # width, height
    position_inf_num = (95, 25)
    position_uninf = (20, 50)
    position_uninf_num = (110, 50)
    position_unk = (20, 75)
    position_unk_num = (100, 75)
    box_position = (10, 10)
    box_position_end = (145, 85) 
    
    number_of_infected = 0
    number_of_uninfected = 0
    total = 0
    unk = 0
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        total += 1
        if predictions[i][0] >= 0.16:
            colour = (0, 255, 0) # Green = uninfected
            number_of_uninfected += 1
        elif predictions[i][0] <= 0.15:
            colour = (0, 0, 255) # Red = infected (probabilistic)
            number_of_infected += 1
        else:
            colour = (255, 0, 0) # Blue = unknown
            unk += 1

        cv2.rectangle(image_with_rois, (x, y), (x + w, y + h), colour, 2)

    cv2.rectangle(image_with_rois, box_position, box_position_end, (50,50,50), -1)
    cv2.putText(image_with_rois, "Infected:", position_inf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image_with_rois, str(number_of_infected), position_inf_num, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image_with_rois, "Uninfected:", position_uninf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image_with_rois, str(number_of_uninfected), position_uninf_num, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image_with_rois, "Unknown:", position_unk, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(image_with_rois, str(unk), position_unk_num, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return image_with_rois, number_of_infected, number_of_uninfected, unk, total

# ================================================================================================================= #
