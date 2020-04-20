import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten, Input
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

my_data_dir="D:\\Work\\Projects\\Covid-19_model\\dataset" #Write Your directory name in which dataset is present

data=[]
labels=[]

# For Covid Data
covid_dir=os.path.join(my_data_dir,os.listdir(my_data_dir)[0])

# For Normal Data
normal_dir=os.path.join(my_data_dir,os.listdir(my_data_dir)[1])

# Saving covid image to data list.
# Changing Color Channels.
# Resizing image to be fixed (224 X 224) pixels.
for r,d,f in os.walk(covid_dir):
    for file in f:
        image=cv2.imread(os.path.join(covid_dir,file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        labels.append(os.listdir(my_data_dir)[0])

# Saving Normal image to data list.
# Changing Color Channels.
# Resizing image to be fixed (224 X 224) pixels.
for r,d,f in os.walk(normal_dir):
    for file in f:
        image=cv2.imread(os.path.join(normal_dir,file))
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        labels.append(os.listdir(my_data_dir)[1])

# Converting both the lists into numpy array and normalizing pixel intensities b/w 0 & 1.
data=np.array(data)/255
labels=np.array(labels)

# One-hot encoding on labels
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

# 80% data for training and 20% data for testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data augmentation object
image_gen = ImageDataGenerator(rotation_range=15, # rotate the image 15 degrees
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

input_shape=X_train[0].shape

# Load the VGG16 model with head Fully Connected layer sets as off
base_model = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# Head model which will be placed on top of base model
head_model = base_model.output
head_model = MaxPooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(64, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# Looping over base model layers so that they are not updated during first training process
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])

# Training
results=model.fit_generator(
            image_gen.flow(X_train,y_train, batch_size=8),
            validation_data=(X_test,y_test),
            epochs=25)

# Predection on test set
predictions=model.predict(X_test)
predictions=np.argmax(predictions,axis=1)

# Classification Report
print(classification_report(y_test.argmax(axis=1), predictions, target_names=lb.classes_))

# Confusion Matrix
cm=confusion_matrix(y_test.argmax(axis=1),predictions)
tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1),predictions).ravel()

# Calculating accuracy, sensitivity, specificity from confusion matrix
accuracy=(tn + tp)/(tn+fp+fn+tp)
sensitivity=tp/(tp+fn)
specificity=tn/(tn+fp)

print("Accuracy: {}".format(accuracy))
print("Sensitivity: {}".format(sensitivity))
print("Specificity: {}".format(specificity))

track=pd.DataFrame(model.history.history)

# Plotting Losses
track[['loss','val_loss']].plot()
plt.xlabel("Epochs")

# Plotting Accuracy
track[['accuracy','val_accuracy']].plot()
plt.xlabel('Epochs')

# Save model
#model.save('covid_initial.h5')
