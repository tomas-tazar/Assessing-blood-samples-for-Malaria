import matplotlib.pyplot as plt, numpy as np
from keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score, f1_score, confusion_matrix, f1_score
from model_training import sequential_training_model, sequential_training_model_v2, train_vgg16

# ========================================================================================================

def plot_model_information(model_trained, epochs):
    print('='*20)
    print('Displaying model information...')
    print('='*20)
        
    acc = model_trained.history['accuracy']
    val_acc = model_trained.history['val_accuracy']
    loss = model_trained.history['loss']
    val_loss = model_trained.history['val_loss']
    epochs_range = range(epochs)
    
    print('-'*20)
    print("Plotting training and validation accuracy and loss...")
    print('-'*20)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, '-o', color='blue', label='Accuracy')
    plt.plot(epochs_range, val_acc, '-o', color='green', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, '-o', color='red', label='Loss')
    plt.plot(epochs_range, val_loss, '-o', color='orange', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss')

    plt.show()

# ========================================================================================================

def plot_metrics(model, testing_data):
    print('='*20)
    print('Plotting Accuracy, Recall, Precision, Average Precision Score and F1 Score for the Sequential model...')
    print('='*20)

    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()

    for batch in testing_data.as_numpy_iterator():
        try:
            x, y = batch
            y_pred = model.predict(x)
            precision.update_state(y, y_pred)
            recall.update_state(y, y_pred)
            accuracy.update_state(y, y_pred)
            precision_score = average_precision_score(y, y_pred)
        except:
            print('Error')
        break
    
    accuracy = accuracy.result().numpy()
    recall = recall.result().numpy()
    precision = precision.result().numpy()
    f1_score = 2 * (precision * recall) / (precision + recall)

    print('Accuracy: ', accuracy) # Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)
    print('Recall: ', recall) # Recall = True Positives / (True Positives + False Negatives)
    print('Precision: ', precision) # Precision = True Positives / (True Positives + False Positives)
    print('Average Precision Score: ', precision_score, '\n') # Average Precision Score = Average of all precision scores at different thresholds
    print("F1-Score: ", f1_score)

# ========================================================================================================

def plot_vgg_metrics(model, testing_data):
    print('='*20)
    print('Plotting Accuracy, Recall, Precision, Average Precision Score and F1 Score for the VGG16 model...')
    print('='*20)
    
    # model = tf.keras.models.load_model('../Dependencies/model_information/models/VGG16_model.h5')
    y_true = testing_data
    y_pred = model.predict(testing_data)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)

    true_labels = []
    for i in range(len(testing_data)):
        _, labels = testing_data[i]
        true_labels.extend(np.argmax(labels, axis=1))

    predicted_labels = []
    for i in range(len(testing_data)):
        images, _ = testing_data[i]
        predictions = model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))

    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('Average Precision Score: ', avg_precision, '\n')
    print("F1 score of the VGG model is: ", f1_score(true_labels, predicted_labels))

# ========================================================================================================

def plot_confusion_matrix(model, testing_data):
    return 0

# ========================================================================================================

def predicting_classes(model, testing_data):
    return 0

# ========================================================================================================

# ROC Curve (Semi useful/ pointless)

# AUC ROC Curve (Not as good as McNemar's test)

# Number of correct detections / Number of detections, Number of True Positives, False Postives, False Nagatives, True Negatives this is the real accuracy of the model (Necessary)

# McNemar's test (should be used to evaluate performance!!!)

# Evaluating performance with FACT (uses McNemar's test)

# MNIST 

# ========================================================================================================

# Driver Code

model, model_trained, epochs, batchesNUM, testing_data = sequential_training_model() # Uncomment to train the first Sequential model optimized for image recognition
# model, model_trained, epochs, batchesNUM, testing_data = sequential_training_model_v2() # Uncomment to train the second Sequential model with Spatial Attention
# model, model_trained, epochs, batchesNUM, testing_data = train_vgg16() # Uncomment to train the state-of-the-art VGG16 model

plot_model_information(model_trained, epochs)
plot_metrics(model, testing_data)
plot_vgg_metrics(model, testing_data)
plot_confusion_matrix(model, testing_data)
predicting_classes(model, testing_data)

