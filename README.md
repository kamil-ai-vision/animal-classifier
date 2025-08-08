# 🐅 Animal Image Classifier

A deep learning model built to classify images into one of **15 animal categories** using **MobileNetV2** and transfer learning with TensorFlow and Keras.  
Achieves **93.47% accuracy** on the validation set.

---

## 🌟 Features

- ✅ **15 Animal Classes**  
  Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra

- 🚀 **High Accuracy**  
  93.47% on validation set

- 🧠 **Transfer Learning**  
  Built on MobileNetV2 pre-trained on ImageNet

- 🛠️ **Complete Workflow**  
  Scripts for training (`train.py`), evaluation (`evaluate.py`), and prediction (`predict.py`)

---

## 🛠️ Setup and Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/animal-classifier.git
cd animal-classifier
```

### 2️⃣ Download Dataset and Model

- Dataset

  📥 [[DOWNLOAD LINK FOR YOUR DATASET]](https://drive.google.com/drive/folders/1aMS4baJ0q-q-w4MI-jJ_NS57H-4_egNo?usp=sharing)

  ⬇️ Unzip and place the dataset in the root directory under dataset/

- Model

  📥 [[DOWNLOAD LINK FOR YOUR .keras MODEL]](https://drive.google.com/drive/folders/1qcVMNqnfym63kw7G7rDZ_WHBYeflTFD6?usp=sharing)

  ⬇️ Place animal_classifier.keras in the models/ folder

### Expected Project Structure:

```
animal-classifier/
├── dataset/
│   ├── Bear/
│   └── ...
├── models/
│   └── animal_classifier.keras
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

### 3️⃣ Set Up Virtual Environment

```
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Use

All scripts should be run from within the src directory.

```
cd src
```

### Evaluate Model Accuracy

To verify the model's accuracy on the validation set:
```
python evaluate.py
```

### Predict a New Image

To classify your own images:

1. Open the `src/predict.py` file in your editor.

2. Find the `images_to_predict` list and change the file paths to point to your images.

3. Run the script without any arguments:
```
python predict.py
```

---

## 📊 Performance and Results

Achieved 93.47% validation accuracy.

| Image File    | Ground Truth | Prediction | Confidence | Result    |
| ------------- | ------------ | ---------- | ---------- | --------- |
| Bear1.jpg     | Bear         | Bear       | 71.69%     | ✅ Correct |
| Bird1.jpg     | Bird         | Bird       | 96.50%     | ✅ Correct |
| Cat3.jpg      | Cat          | Cat        | 98.56%     | ✅ Correct |
| Cow3.jpg      | Cow          | Cow        | 99.38%     | ✅ Correct |
| Deer2.jpg     | Deer         | Deer       | 53.98%     | ✅ Correct |
| Dog4.jpg      | Dog          | Dog        | 80.84%     | ✅ Correct |
| Dolphin1.jpg  | Dolphin      | Dolphin    | 98.71%     | ✅ Correct |
| Elephant5.jpg | Elephant     | Elephant   | 99.53%     | ✅ Correct |
| Giraffe2.jpg  | Giraffe      | Giraffe    | 99.64%     | ✅ Correct |
| Horse1.jpg    | Horse        | Horse      | 88.89%     | ✅ Correct |
| Kangaroo4.jpg | Kangaroo     | Kangaroo   | 92.27%     | ✅ Correct |
| Lion2.jpg     | Lion         | Lion       | 98.54%     | ✅ Correct |
| Panda5.jpg    | Panda        | Panda      | 99.75%     | ✅ Correct |
| Tiger2.jpg    | Tiger        | Tiger      | 99.97%     | ✅ Correct |
| Zebra1.jpg    | Zebra        | Zebra      | 99.11%     | ✅ Correct |
