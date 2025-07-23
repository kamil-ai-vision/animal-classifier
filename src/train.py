import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
# Use relative paths to make the script portable
BASE_DATASET_PATH = '../dataset/' # Path relative to the src/ folder
MODEL_SAVE_PATH = '../models/animal_classifier_v2.keras' # Save new models with a different name

def train_model():
    """
    This function sets up the data generators, builds, compiles,
    and trains the animal classification model.
    """
    # --- 1. Data Generators ---
    print("Setting up data generators...")

    # We use validation_split directly on the main dataset folder.
    # This is simpler than manually splitting files into train/val folders.
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Automatically reserve 20% of data for validation
    )

    # Training Generator
    train_generator = datagen.flow_from_directory(
        BASE_DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training', # Specify this is the training set
        shuffle=True
    )

    # Validation Generator
    validation_generator = datagen.flow_from_directory(
        BASE_DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation', # Specify this is the validation set
        shuffle=False
    )

    # --- 2. Model Building (Transfer Learning) ---
    print("Building the model...")
    # Load MobileNetV2 with pre-trained ImageNet weights, excluding the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers so they are not updated during initial training
    base_model.trainable = False

    # Add our custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Pool the features to a single vector
    x = Dropout(0.5)(x) # Add dropout for regularization
    x = Dense(128, activation='relu')(x)
    # The final layer has one neuron per class (15 in your case)
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # --- 3. Compile the Model ---
    print("Compiling the model...")
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --- 4. Callbacks ---
    # Ensure the target directory for the model exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # Stop training early if validation loss doesn't improve
    early_stop = EarlyStopping(
        patience=5,
        monitor='val_loss',
        restore_best_weights=True,
        verbose=1
    )

    # --- 5. Train the Model ---
    print("Starting model training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    print(f"\n✅ Training complete. Best model saved to {MODEL_SAVE_PATH}")
    return history

def plot_history(history):
    """Plots the training and validation accuracy and loss."""
    # Accuracy Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../training_plots.png') # Save the plots as an image
    plt.show()


if __name__ == "__main__":
    training_history = train_model()
    if training_history:
        plot_history(training_history)