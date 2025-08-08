import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2 # Switched to ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import json

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# Epochs for each phase
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
# Use relative paths to make the script portable
BASE_DATASET_PATH = 'D:/VS Code/animal-classifier/dataset/'
MODEL_SAVE_PATH = 'D:/VS Code/animal-classifier/models/animal_classifier_v3.keras'
CLASS_INDICES_PATH = 'D:/VS Code/animal-classifier/models/class_indices.json'

def train_model():
    """
    This function sets up the data generators, builds, compiles,
    and trains the animal classification model in two phases.
    """
    # --- 1. Data Generators ---
    print("Setting up data generators...")

    # Data Augmentation for the TRAINING set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Reserve 20% of data for validation
    )

    # NO Data Augmentation for the VALIDATION set, only rescaling
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        BASE_DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        BASE_DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Create the 'models' directory if it doesn't exist
    os.makedirs(os.path.dirname(CLASS_INDICES_PATH), exist_ok=True)
    
    # Save class indices for consistent evaluation and prediction
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"✅ Class indices saved to {CLASS_INDICES_PATH}")

    # --- 2. Model Building (Transfer Learning) ---
    print("Building the base model (ResNet50V2)...")
    # Switched from MobileNetV2 to ResNet50V2
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Add our custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x) # Increased neurons in the dense layer
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.summary()

    # --- 3. Phase 1: Train the custom head only ---
    print("\n--- Phase 1: Training custom classification head ---")
    model.compile(
        optimizer=Adam(learning_rate=0.001), # Higher learning rate for the new layers
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for phase 1
    early_stop_p1 = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True, verbose=1)

    history_p1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS,
        callbacks=[early_stop_p1]
    )

    # --- 4. Phase 2: Fine-Tuning ---
    print("\n--- Phase 2: Fine-Tuning base model layers ---")
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze all layers except for the last 'n' layers
    for layer in base_model.layers[:-30]:  # Example: Unfreeze the last 30 layers
        layer.trainable = False
        
    # Recompile the model with a much lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001), # Very low learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for phase 2
    early_stop_p2 = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001, verbose=1)

    history_p2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS, # Total epochs will be INITIAL + FINE_TUNE
        initial_epoch=history_p1.epoch[-1],
        callbacks=[checkpoint, early_stop_p2, reduce_lr]
    )

    print(f"\n✅ Training complete. Best model saved to {MODEL_SAVE_PATH}")
    return history_p1, history_p2

def plot_history(history_p1, history_p2):
    """Plots the training and validation accuracy and loss for both phases."""
    # Combine histories for a single plot
    acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
    val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
    loss = history_p1.history['loss'] + history_p2.history['loss']
    val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']
    epochs = range(len(acc))

    # Accuracy Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../training_plots.png')
    plt.show()

if __name__ == "__main__":
    training_history = train_model()
    if training_history:
        history_phase1, history_phase2 = training_history
        plot_history(history_phase1, history_phase2)