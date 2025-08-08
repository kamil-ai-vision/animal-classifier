# 🐅 Animal Image Classifier

A deep learning model built to classify images into one of **15 animal categories** using **ResNet50V2** and **transfer learning** with TensorFlow and Keras.

---

## 🌟 Features

- ✅ **15 Animal Classes**  
  Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra

- 🚀 **High Accuracy**  
  Achieved **91.91%** on the validation set

- 🧠 **Transfer Learning**  
  Built on **ResNet50V2** pre-trained on ImageNet

- 🛠️ **Complete Workflow**  
  Includes scripts for:
  - Training: `train.py`
  - Evaluation: `evaluate.py`
  - Prediction: `predict.py`

---

## 🛠️ Setup and Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kamil-ai-vision/animal-classifier.git
cd animal-classifier
```

### 2️⃣ Download Dataset and Model

#### 📦 Dataset  
📥 [Download Dataset](https://drive.google.com/drive/folders/1aMS4baJ0q-q-w4MI-jJ_NS57H-4_egNo?usp=sharing)  
⬇️ Unzip and place the dataset in the root directory under `dataset/`

#### 🧠 Model  
📥 [Download `.keras` Model](https://drive.google.com/file/d/14EeLhg_oBVpryk69V3qzX2sx2qn7U_Zu/view?usp=sharing)  
⬇️ Place `animal_classifier.keras` inside the `models/` folder

#### 📁 Expected Project Structure

```
animal-classifier/
├── dataset/
│ ├── Bear/
│ └── ...
├── models/
│ └── animal_classifier.keras
├── src/
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

### 3️⃣ Set Up Virtual Environment

```bash
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

> 📁 All scripts should be run from within the `src` directory.

```bash
cd src
```

### 🧪 Evaluate Model Accuracy

To verify the model's accuracy on the validation set:

```
python evaluate.py
```

### Predict a New Image

To classify your own images:
1. Open the `src/predict.py` file in your editor.
2. Find the  `images_to_predict` list and change the file paths to point to your images.
3. Run the script without any arguments:
   ```
   python predict.py
   ```

---

## 📊 Performance and Results

The model achieved **91.91% validation accuracy** using a **ResNet50V2** backbone.

---

### 🔍 Detailed Metrics
```
---------- Overall Evaluation ----------
📊 Validation Loss: 0.2784
🎯 Validation Accuracy: 91.91%
---------------------------------------
```
```
---------- Detailed Metrics ----------
📋 Classification Report:
              precision    recall  f1-score   support

        Bear       1.00      0.96      0.98        25
        Bird       0.92      0.85      0.88        27
         Cat       1.00      0.96      0.98        24
         Cow       0.81      1.00      0.90        26
        Deer       0.96      0.92      0.94        25
         Dog       0.83      0.79      0.81        24
     Dolphin       0.95      0.80      0.87        25
    Elephant       0.96      1.00      0.98        26
     Giraffe       0.89      1.00      0.94        25
       Horse       0.72      0.81      0.76        26
    Kangaroo       1.00      0.92      0.96        25
        Lion       0.86      0.92      0.89        26
       Panda       0.96      1.00      0.98        27
       Tiger       1.00      0.88      0.94        25
       Zebra       1.00      0.96      0.98        27

    accuracy                           0.92       383
   macro avg       0.92      0.92      0.92       383
weighted avg       0.92      0.92      0.92       383
```

### 📈 Confusion Matrix:
```
[24  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
[ 0 23  0  0  0  0  1  0  0  3  0  0  0  0  0]
[ 0  0 23  1  0  0  0  0  0  0  0  0  0  0  0]
[ 0  0  0 26  0  0  0  0  0  0  0  0  0  0  0]
[ 0  0  0  0 23  0  0  0  0  2  0  0  0  0  0]
[ 0  0  0  0  1 19  0  0  0  0  0  3  1  0  0]
[ 0  0  0  1  0  0 20  0  2  2  0  0  0  0  0]
[ 0  0  0  0  0  0  0 26  0  0  0  0  0  0  0]
[ 0  0  0  0  0  0  0  0 25  0  0  0  0  0  0]
[ 0  0  0  4  0  0  0  0  0 21  0  1  0  0  0]
[ 0  1  0  0  0  0  0  0  0  1 23  0  0  0  0]
[ 0  0  0  0  0  1  0  1  0  0  0 24  0  0  0]
[ 0  0  0  0  0  0  0  0  0  0  0  0 27  0  0]
[ 0  0  0  0  0  3  0  0  0  0  0  0  0 22  0]
[ 0  0  0  0  0  0  0  0  1  0  0  0  0  0 26]
```

---

### Prediction Results

The table below shows the model's predictions on a sample of images.

```
| Image File    | Ground Truth | Prediction | Confidence | Result    |
| ------------- | ------------ | ---------- | ---------- | --------- |
| Bear1.jpg     | Bear         | Bear       | 71.69%     | ✅ Correct|
| Bird1.jpg     | Bird         | Bird       | 96.50%     | ✅ Correct|
| Cat3.jpg      | Cat          | Cat        | 98.56%     | ✅ Correct|
| Cow3.jpg      | Cow          | Cow        | 99.38%     | ✅ Correct|
| Deer2.jpg     | Deer         | Deer       | 53.98%     | ✅ Correct|
| Dog4.jpg      | Dog          | Dog        | 80.84%     | ✅ Correct|
| Dolphin1.jpg  | Dolphin      | Dolphin    | 98.71%     | ✅ Correct|
| Elephant5.jpg | Elephant     | Elephant   | 99.53%     | ✅ Correct|
| Giraffe2.jpg  | Giraffe      | Giraffe    | 99.64%     | ✅ Correct|
| Horse1.jpg    | Horse        | Horse      | 88.89%     | ✅ Correct|
| Kangaroo4.jpg | Kangaroo     | Kangaroo   | 92.27%     | ✅ Correct|
| Lion2.jpg     | Lion         | Lion       | 98.54%     | ✅ Correct|
| Panda5.jpg    | Panda        | Panda      | 99.75%     | ✅ Correct|
| Tiger2.jpg    | Tiger        | Tiger      | 99.97%     | ✅ Correct|
| Zebra1.jpg    | Zebra        | Zebra      | 99.11%     | ✅ Correct|
```

---

## 👤 Author

**Kamil** — [GitHub Profile](https://github.com/kamil-ai-vision)
