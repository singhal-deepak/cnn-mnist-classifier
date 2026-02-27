# CNN MNIST Digit Classification

## Overview
This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

src/
model.py — Configurable CNN architecture  
train.py — Training pipeline  
tune.py — Hyperparameter experiments  
predict.py — Evaluation & inference  
utils.py — Data loading & preprocessing  

## Hyperparameter Experiments

Tested variations in:
- Number of filters
- Kernel size
- Dropout rate
- Learning rate
- Batch normalization

Best accuracy achieved: ~98.7%

## How to Run

Install dependencies:

pip install -r requirements.txt

Train model:

python src/train.py

Run hyperparameter tuning:

python src/tune.py

Evaluate model:

python src/predict.py

## Key Learnings
- Larger kernel sizes improved feature capture.
- Learning rate impacted convergence stability.
- Increasing filters showed diminishing returns.
- Controlled experiments are essential for architecture selection.