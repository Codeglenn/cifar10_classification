from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = keras.models.load_model('saved_model/cifar10_model.keras')

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# List of image file paths (add your image filenames here)
image_files = ['images/horse.jpg', 'images/cat.jpg', 'images/dog.jpg' , 'images/frog.jpg',
                'images/ship.jpg', 'images/airplane.jpg', 'images/automobile.jpg', 
                'images/bird.jpg', 'images/deer.jpg', 'images/truck.jpg']  

for img_path in image_files:
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue

    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_class = np.argmax(pred[0])

    print(f"\nImage: {img_path}")
    print("Prediction probabilities:", pred[0])
    print("Predicted class:", class_names[predicted_class])

    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()

