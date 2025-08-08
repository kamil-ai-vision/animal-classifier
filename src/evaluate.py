import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# Use relative paths and the new model name
DATASET_PATH = 'D:/VS Code/animal-classifier/dataset/'
MODEL_PATH  = 'D:/VS Code/animal-classifier/models/animal_classifier_v3.keras'
CLASS_INDICES_PATH = 'D:/VS Code/animal-classifier/models/class_indices.json'

def evaluate_model():
    """
    Loads the saved model and evaluates its performance on the validation set
    with detailed metrics.
    """
    # 1. Check if model and class indices files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"❌ Error: Class indices file not found at {CLASS_INDICES_PATH}")
        return

    print(f"✅ Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class indices
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Get class labels from the loaded dictionary
    class_labels = list(class_indices.keys())

    # 2. Set up the validation data generator
    # IMPORTANT: Only rescaling, no data augmentation for evaluation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    validation_generator = validation_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False  # Crucial for matching predictions to true labels
    )

    # 3. Evaluate the model and get predictions for detailed metrics
    print("\n🔬 Evaluating model performance on the validation dataset...")
    results = model.evaluate(validation_generator, verbose=0)
    
    # Get true labels and predictions
    validation_generator.reset()
    y_pred_probs = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = validation_generator.classes

    # 4. Print the results
    loss = results[0]
    accuracy = results[1]

    print("\n---------- Overall Evaluation ----------")
    print(f"📊 Validation Loss: {loss:.4f}")
    print(f"🎯 Validation Accuracy: {accuracy * 100:.2f}%")
    print("---------------------------------------")

    print("\n---------- Detailed Metrics ----------")
    print("📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

    print("📈 Confusion Matrix:")
    # Pretty-print the confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    for row in cm:
        print(row)
    print("---------------------------------------\n")


if __name__ == "__main__":
    evaluate_model()