# Tamil Character Image Classification

## Overview
This project aims to classify Tamil character images by predicting their vowel and consonant components using Convolutional Neural Networks (CNNs). The goal is to develop a robust deep learning model that accurately identifies these components from character images.

## Dataset
The dataset consists of Tamil character images labeled with their corresponding vowel and consonant components. The images are preprocessed and augmented to enhance model performance.

## Approach
- **Image Preprocessing:** Resizing, normalization, and data augmentation (flipping, rotation, etc.).
- **Model Architecture:** Utilized CNN-based architectures such as ResNet, and custom CNN models.
- **Training Strategy:** Applied transfer learning, dropout regularization, batch normalization, and learning rate scheduling.
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score.

## Model Performance
- **Achieved Kaggle leaderboard score:** 0.91970 (private and public).
- **Optimized using:** Hyperparameter tuning, Adam optimizer, and learning rate scheduling.

## Tools & Technologies
- **Deep Learning Frameworks:** PyTorch, Torchvision
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Model Evaluation:** Scikit-learn

## Results & Insights
- The model effectively learns to distinguish Tamil vowels and consonants from character images.
- Data augmentation significantly improved model generalization.
- Hyperparameter tuning played a crucial role in achieving high accuracy.

## Future Work
- Experimenting with Vision Transformers for improved accuracy.
- Expanding the dataset with more character variations.
- Deploying the model as an API for real-time classification.

