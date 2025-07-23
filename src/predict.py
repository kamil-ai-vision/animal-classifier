import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2 # OpenCV for image processing
import os

# --- Main Prediction Function (This function remains the same) ---
def predict_animal(model, image_path):
    """
    Loads a trained model and an image, and predicts the animal class.

    Args:
        model (tf.keras.Model): The loaded Keras model.
        image_path (str): Path to the input image.
    """
    try:
        # 1. Define constants that must match your training setup
        IMAGE_SIZE = (224, 224)
        CLASS_NAMES = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
                       'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
                       'Panda', 'Tiger', 'Zebra']

        # 2. Load and preprocess the image
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found at {image_path}")
            return

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
        
        # Normalize pixel values to [0, 1] and add a batch dimension
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        # 3. Make prediction
        prediction = model.predict(img_array)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100

        print(f"File: {os.path.basename(image_path)}")
        print(f"🐾 Predicted Animal: {predicted_class_name}")
        print(f"🎯 Confidence: {confidence:.2f}%\n")

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}\n")


# --- Main execution block ---
if __name__ == "__main__":
    # Define the path to your model
    model_path = "../models/animal_classifier.keras"
    
    # --- 🖍️ EDIT THIS LIST ---
    # Add the paths to the images you want to predict.
    # Use relative paths from the 'src' folder.
    images_to_predict = [
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Bear\Bear1.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Bird\Bird1.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Cat\Cat3.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Cow\Cow3.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Deer\Deer2.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Dog\Dog4.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Dolphin\Dolphin1.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Elephant\Elephant5.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Giraffe\Giraffe2.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Horse\Horse1.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Kangaroo\kangaroo4.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Lion\Lion2.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Panda\Panda5.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Tiger\Tiger2.jpg",
        r"C:\Users\admin\OneDrive\Desktop\Unified Mentor\Animal Image Classification\Animals\Zebra\Zebra1.jpg"
    ]
    # -------------------------

    # Load the model once
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
    else:
        print("✅ Loading model...")
        model = load_model(model_path)
        print("---------------------------------------\n")
        
        # Loop through the list and predict each image
        for image_path in images_to_predict:
            predict_animal(model, image_path)