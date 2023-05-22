import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false'
uid = 1000
os.environ['XDG_RUNTIME_DIR'] = f'/run/user/{os.getuid()}'
import cv2
import pandas as pd
import numpy as np

from PyQt5.QtCore import QLibraryInfo

os.environ['QT_PLUGIN_PATH'] = os.path.join(QLibraryInfo.location(QLibraryInfo.PluginsPath))


# # Get a list of all image file names in the folder
galaxy_df = pd.read_csv("/home/cyrus/Documents/galaxy_df_final.csv")

# Set the path to the directory containing the images
image_directory = "/home/cyrus/Downloads/images_gz2/images"

# Initialize empty lists to store the matched image file names and their corresponding 'gz2class' values
matched_images = []
matched_labels = []

# Iterate through the rows in the table
for index, row in galaxy_df.iterrows():

    if len(matched_images) >= 5000:
        break
    
    # Extract the 'id' value
    image_id = row["asset_id"]

    # Create the corresponding image file name
    image_file = f"{image_id}.jpg"

    # Check if the image file exists in the directory
    if os.path.isfile(os.path.join(image_directory, image_file)):
        # Add the image file name to the matched_images list
        matched_images.append(image_file)
        # Add the corresponding 'gz2class' value to the matched_labels list
        matched_labels.append(row["gz2class"])

# Set the desired dimensions for resizing
resize_dim = (128, 128)

# Set the batch size
batch_size = 10

# Initialize an empty list to store the batches of images and labels
image_batches = []
label_batches = []

# Loop through the image files and labels in batches
images = []
for i in range(0, len(matched_images), batch_size):
    batch_images = matched_images[i:i + batch_size]
    batch_labels = matched_labels[i:i + batch_size]
    image_arrays = []

    for image_file in batch_images:
        image_path = os.path.join(image_directory, image_file)

        # Read the image from file
        image = cv2.imread(image_path)

        if image is not None:
            # Convert the image from BGR to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize the image
            resized_image = cv2.resize(image, resize_dim)

            # Append the resized image to the list of image arrays
            images.append(resized_image)

    # Add the batch of image arrays and labels to their respective lists
    # image_batches.append(image_arrays)
    label_batches.extend(batch_labels)

# Convert the list of images and labels to NumPy arrays
# images_array = np.array(image_batches)
# labels_array = np.array(label_batches)

X = np.array(images)
y = np.array(label_batches)


# Normalize pixel values
X = X.astype('float32')
X /= 255.0

from sklearn.preprocessing import LabelEncoder

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Assuming your data is in the following format
# X = np.array(images_array)  # Images as numpy arrays
# y = np.array(matched_labels)       # Labels as integer values

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size = 0.2, random_state = 42, stratify = y_encoded)



from tensorflow.keras import layers, models
from keras_tuner import HyperModel

class GalacticHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = models.Sequential()
        
        # Convolutional and pooling layers with tunable hyperparameters
        model.add(layers.Conv2D(filters=hp.Int('filters_1', 16, 64, step = 16),
                                kernel_size=(3, 3),
                                activation = 'relu',
                                input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(filters=hp.Int('filters_2', 32, 128, step = 32),
                                kernel_size=(3, 3),
                                activation = 'relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(filters=hp.Int('filters_3', 64, 256, step = 64),
                                kernel_size=(3, 3),
                                activation = 'relu'))

        # Fully connected layers
        model.add(layers.Flatten())
        model.add(layers.Dense(units=hp.Int('dense_units', 32, 256, step = 32),
                               activation = 'relu'))
        model.add(layers.Dense(self.num_classes, activation = 'softmax'))

        # Compile the model with tunable learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling = 'log'))
        # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

        return model









from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

input_shape = (128, 128, 3)
num_classes = 3

# Instantiate the HyperModel
galactic_hypermodel = GalacticHyperModel(input_shape, num_classes)

# Set up the RandomSearch tuner
tuner = RandomSearch(galactic_hypermodel,
                     objective = 'val_accuracy',
                     max_trials = 30,  # Number of different hyperparameter combinations to try
                     seed = 42,
                     project_name = 'galaxy_cnn_tuning')

# Search for the best hyperparameters
tuner.search(X_train, y_train,
             epochs = 50,
             validation_data = (X_val, y_val),
             callbacks = [tf.keras.callbacks.EarlyStopping(patience=5)])

# Get the best model found by the tuner
best_hp = tuner.get_best_hyperparameters()[0]
galaxy_cnn = tuner.hypermodel.build(best_hp)


# Set hyperparameters
batch_size = 32
epochs = 50
learning_rate = 0.0001

# Compile the model with the chosen learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
galaxy_cnn.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = galaxy_cnn.fit(X_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(X_val, y_val),
                         callbacks=[early_stopping])



import pickle

with open("history.pickle", "wb") as f:
    pickle.dump(history.history, f)


print('Finished with the model!')





