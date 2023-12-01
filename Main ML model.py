# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Step 1.1: Data Collection
def collect_data(data_dir):
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            images.append(img)
    return images

# Step 1.2: Data Labeling
def label_data(images):
    labels = []
    for img in images:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Is this image good (1) or bad (0)?")
        plt.show()
        label = int(input("Enter label (0 for bad, 1 for good): "))
        labels.append(label)
    return labels

# Example usage:
data_directory = "/kaggle/input/cable-connector-anomaly-detection"
images = collect_data(data_directory)
labels = label_data(images)

# Create a DataFrame to store the image paths and corresponding labels
df = pd.DataFrame({"Image_Path": [os.path.join(data_directory, f) for f in os.listdir(data_directory)],
                   "Label": labels})

# Save the DataFrame to a CSV file for future reference
df.to_csv("dataset_labels.csv", index=False)



import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 2.1: Image Resizing
def resize_images(images, target_size=(224, 224)):
    resized_images = [cv2.resize(img, target_size) for img in images]
    return resized_images

# Step 2.2: Normalization
def normalize_images(images):
    normalized_images = [img / 255.0 for img in images]  # Normalize pixel values to [0, 1]
    return normalized_images

# Load the dataset and labels (replace with your actual dataset and labels)
df = pd.read_csv("dataset_labels.csv")
image_paths = df["Image_Path"].tolist()
labels = df["Label"].tolist()

# Load images
images = [cv2.imread(img_path) for img_path in image_paths]

# Step 2.1: Resize Images
resized_images = resize_images(images)

# Step 2.2: Normalize Images
normalized_images = normalize_images(resized_images)
print(labels)
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(normalized_images, labels, test_size=0.3, random_state=42)

# Save the preprocessed data (optional)
np.save("X_train.npy", np.array(X_train))
np.save("X_val.npy", np.array(X_val))
np.save("y_train.npy", np.array(y_train))
np.save("y_val.npy", np.array(y_val))
print(df.head(5))



import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Save the test data
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Model Development
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification output

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Test Data
# Load and preprocess test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Optional: Save the trained model
model.save("good_bad_classifier_model.keras")



# Step 3.2: Split Data
input_shape = X_test[0].shape
model = create_model(input_shape)

# Step 3.3: Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3.4: Model Training
history = model.fit(X_test, y_test, epochs=10, validation_data=(X_val, y_val))

# Optional: Save the trained model
model.save("good_bad_classifier_model.keras")





import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess a single image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match the input size used during training
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Replace "path/to/your/image.jpg" with the actual path to your input image
input_image_path = '/kaggle/input/cable-connector-anomaly-detection/img1.png'
# Preprocess the input image
input_image = preprocess_image(input_image_path)

# Perform prediction
prediction = model.predict(input_image)

# Interpret the prediction (assuming binary classification)
class_label = "Good" if prediction[0][0] > 0.5 else "Bad"

print(f"The classification result for the input image is: {class_label}")




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 5.1: Metrics
y_pred = model.predict(X_val)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

accuracy = accuracy_score(y_val, y_pred_binary)
precision = precision_score(y_val, y_pred_binary)
recall = recall_score(y_val, y_pred_binary)
f1 = f1_score(y_val, y_pred_binary)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_binary)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Bad (0)", "Good (1)"],
            yticklabels=["Bad (0)", "Good (1)"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Step 5.2: Adjustments (if necessary)
# If the model's performance is not satisfactory, consider adjusting model architecture or training parameters.




import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
loaded_model = tf.keras.models.load_model("good_bad_classifier_model.keras")

# Assuming you have a small set of test data (replace with actual test data)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Make predictions
y_pred_test = loaded_model.predict(X_test)
y_pred_test_binary = (y_pred_test > 0.5).astype(int)

# Evaluate performance on the test set
accuracy_test = accuracy_score(y_test, y_pred_test_binary)
precision_test = precision_score(y_test, y_pred_test_binary)
recall_test = recall_score(y_test, y_pred_test_binary)
f1_test = f1_score(y_test, y_pred_test_binary)

print("Performance on the Test Set:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
