import keras
import numpy as np
import os
import pandas as pd
# this file is used to make the final predictions on the test dataset for each of the four models. The paths need to be
# set accordingly. It also computes the accuracy on the test dataset.

# dimensions of our images
img_width, img_height = 299, 299

# load the model we saved
model = keras.models.load_model('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Social_Model.keras')
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['categorical_accuracy'])
# get list of images and make predictions:
path = 'C:/Users/ferdi/Social_test'
image_name = os.listdir(path)
predictions = []
correct = 0
for name in image_name:
    image_path = os.path.join(path, name)
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    y_prob = model.predict(images, batch_size=10)
    y_class_int = np.argmax(y_prob, axis=1)
    if y_class_int == 0:
        y_class = 'Rural'
    elif y_class_int == 1:
        y_class = 'Urban'

    if y_class in name:
        correct += 1
        classification_correct = True
    else:
        classification_correct = False
    max_prob = max(max(y_prob))
    if max_prob < 0.7:
        delegate = True
    else:
        delegate = False
    single_pred = [name, y_prob, y_class, classification_correct, delegate]
    predictions.append(single_pred)

accuracy = correct / len(image_name)
print(accuracy)
pred_df = pd.DataFrame(predictions, columns=['Filename', 'Probabilities', 'Predicted_Class', 'Correctly_Classified', 'Delegate'])
pred_df.to_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Social_Predictions.xlsx')
