import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# Use relative paths
MODEL_PATH = '../models/animal_classifier.keras'
DATASET_PATH = '../dataset/'

def evaluate_model():
    """
    Loads the saved model and evaluates its performance on the validation set.
    """
    # 1. Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("Please ensure your trained model is in the 'models' directory.")
        return

    print(f"✅ Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Set up the validation data generator
    # IMPORTANT: The preprocessing (rescale=1./255) must be the same as during training.
    # We use validation_split to carve out the same 20% of data for validation.
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # Must match the split used in training
    )

    validation_generator = validation_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',  # Specify this is the validation set
        shuffle=False  # No need to shuffle for evaluation
    )

    # 3. Evaluate the model
    print("\n🔬 Evaluating model performance on the validation dataset...")
    results = model.evaluate(validation_generator)

    # 4. Print the results
    loss = results[0]
    accuracy = results[1]

    print("\n---------- Evaluation Complete ----------")
    print(f"📊 Validation Loss: {loss:.4f}")
    print(f"🎯 Validation Accuracy: {accuracy * 100:.2f}%")
    print("---------------------------------------")


if __name__ == "__main__":
    evaluate_model()