import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

model_path = "D:\\lenovo file new\\Project\\first_installation\\controllers\\Concatination_controller\\line_following_model_1_more_accurate_dataset.h5"
model = load_model(model_path)

# Define the dataset path
dataset_path = "D:\\lenovo file new\\Project\\first_installation\\More accurate dataset with its trained model for line following with LSTM\\Database"

# Function to load data
def load_data(dataset_path):
    images = []
    labels = []
    label_dict = {"SF": 0, "SHTL": 1, "SHTR": 2, "SLTL": 3, "SLTR": 4, "SSTL": 5, "SSTR": 6}
    for label, index in label_dict.items():
        folder_path = os.path.join(dataset_path, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((100, 100))
            img_array = np.array(img)
            images.append(img_array)
            labels.append(index)
    return np.array(images), np.array(labels)

# Load test data
X, y = load_data(dataset_path)
X = X / 255.0
y = to_categorical(y, num_classes=7)
X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], X_test.shape[3]))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

# Print the accuracy
print("Model Accuracy: {:.2f}%".format(accuracy * 100))
