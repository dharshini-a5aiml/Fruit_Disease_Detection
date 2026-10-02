# Fruit Disease Detection using AI/ML

A web application that predicts the disease of an apple, guava or mango from an uploaded photo, and shows the confidence of the prediction.

## Problem Statement
Fruit diseases reduce quality and market value, and identifying them by eye needs expertise. This project uses deep learning to classify common fruit diseases from a single image.

## Objectives
- Classify diseased and healthy fruit images using a CNN
- Build a simple web interface for uploading an image and viewing the result
- Display the predicted disease with a confidence percentage

## Technologies Used
Python, TensorFlow/Keras (MobileNetV2), Flask, HTML, CSS, scikit-learn, Matplotlib, Git/GitHub

## System Workflow
User -> Flask web page -> Image upload -> Resize to 224x224 -> Trained MobileNetV2 model -> Prediction -> Disease + confidence shown on the page

## Dataset
"Fruits Dataset for Fruit Disease Classification" from Kaggle:
https://www.kaggle.com/datasets/ateebnoone/fruits-dataset-for-fruit-disease-classification

Only the Apple, Guava and Mango folders were used (12 classes). `prepare_data.py` splits the images into train (80%) and test (20%) folders, using at most 400 images per class. The dataset is not included in this repository.

Classes:
- Apple: Blotch, Healthy, Rot, Scab
- Guava: Anthracnose, Fruitfly, Healthy
- Mango: Alternaria, Anthracnose, Black Mould Rot, Healthy, Stem End Rot

## Model
- Transfer learning with MobileNetV2 (pretrained on ImageNet)
- Data augmentation (flip, rotation, zoom), class weights for imbalanced classes
- Two-stage training: top layers first, then fine-tuning the last 30 layers
- Test accuracy: **86.71%** on 286 unseen test images

## Features
- Upload a JPG or PNG image and preview it
- Predicted fruit, disease and confidence percentage
- Low-confidence warning for unclear or unrelated images
- Handles invalid file types, missing files and a missing model file

## Installation
```
git clone https://github.com/dharshini-a5aiml/Fruit_Disease_Detection.git
cd Fruit_Disease_Detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
The trained model is in the `model/` folder, so you can start the app directly:
```
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

To retrain, download the dataset into `dataset/raw/` (APPLE, GUAVA, MANGO folders), then run:
```
python prepare_data.py
python train.py
```

## How to Use
1. Open the web page
2. Choose a fruit image (JPG or PNG)
3. Click Predict
4. View the disease and confidence

## Sample Output
Add a screenshot of your result page here.

## Limitations
- Only apple, guava and mango, 12 classes in total
- Small dataset: Apple and Guava classes have about 100 images each
- Some similar-looking diseases are confused (for example Scab_Apple and Black Mould Rot in Mango had lower scores)
- The model has no "not a fruit" class, so unrelated images still get a prediction with a low confidence score
- Trained on dataset images, so phone photos with different backgrounds may be less accurate

## Future Enhancements
- Add more fruits and more images per class
- Add a "not a fruit" class
- Add disease treatment suggestions
- Deploy the app online