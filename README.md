# Organic-VS-Recycle-Classification-Using-Transfer-Learning

## Fine-Tuned VGG16 for O vs R Classification

This repository contains a fine-tuned VGG16-based CNN model for binary image classification (O vs R). The project includes:

- **Feature Extraction Model** (`extract_features_vgg16.keras`)
- **Fine-Tuned Model** (`fine_tuned_vgg16.keras`)

## 📌 Project Overview

The project leverages **Transfer Learning** with **VGG16** to classify images into two categories: **O** and **R**.

- **Feature Extraction Model**: Uses pre-trained VGG16 (without fine-tuning).
- **Fine-Tuned Model**: Further trains the last convolutional layers to improve accuracy.

## 🏧 Model Architecture

- **Pretrained VGG16** (ImageNet weights, `include_top=False`)
- **Fully connected layers** (Dense, Dropout, ReLU activations)
- **Binary Classification** (Sigmoid activation for final output)

## 🛠️ Setup & Requirements

Install dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## 🚀 Training & Fine-Tuning

```python
model.fit(train_generator,
          steps_per_epoch=5,
          epochs=10,
          validation_data=val_generator,
          validation_steps=max(1, len(val_generator) // batch_size))
```

## 💊 Model Performance

| Model                 | Accuracy | Precision (O, R) | Recall (O, R) | F1-Score (O, R) |
|-----------------------|----------|------------------|--------------|--------------|
| Feature Extraction   | 84%      | 0.81, 0.87       | 0.88, 0.80   | 0.85, 0.83   |
| Fine-Tuned Model     | 82%      | 0.82, 0.82       | 0.82, 0.82   | 0.82, 0.82   |

## 📂 Model Files

| File                         | Description                                      |
|------------------------------|--------------------------------------------------|
| `extract_features_vgg16.keras` | Feature extraction model (VGG16 frozen layers)  |
| `fine_tuned_vgg16.keras`      | Fine-tuned VGG16 model (last layers unfrozen)   |

## 📌 How to Use

### 1️⃣ Load the Pretrained Model

```python
import tensorflow as tf
model = tf.keras.models.load_model('fine_tuned_vgg16.keras')
```

### 2️⃣ Make Predictions

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('sample.jpg', target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print("Predicted Class:", "O" if prediction < 0.5 else "R")
```

## 🛠 Future Improvements

- Experiment with other CNN architectures (ResNet, EfficientNet, etc.).
- Implement data augmentation for better generalization.
- Deploy the model using Flask/FastAPI for real-time predictions.

## 🐝 License

This project is licensed under the **MIT License**.

## 🔗 Author: **Chaitanya Thombare**

🔥 If you found this helpful, **star ⭐ the repo!**
