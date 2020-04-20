# Covid-19-Detector

## Description
It is very much evident what Deep Learning can achieve in the Medical field, hence in view of the current situation of pandemic and for education purpose, I have created a Web Application that could detect Covid-19 in X-ray images.

## Installation
pip install the following libraries:
   1) pandas
   2) numpy
   3) scikit-learn
   4) tensorflow
   5) keras
   6) flask
   7) opencv-python
   8) Pillow

Dataset: 1) [link for Covid-19 X-ray images](https://github.com/ieee8023/covid-chestxray-dataset)
         2) [link for Healthy persons X-ray images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

Above datasets can be found in this repository in the Covid-19_model/covid-chestxray-dataset-master folder (For Covid-19 patients) and in the Covid-19_model/chest_xray folder (For Healthy persons).

## Usage
covid_positive_dataset.py and kaggle_normal.py files are used to extract images from the above mentioned datasets folder and copy it in Covid-19_model/dataset folder. 

For running the Web application, run the predict_app.py (Flask application) file in the flask_app folder.

![Covid_19_GIF](https://user-images.githubusercontent.com/60289706/79780034-f04a8e00-8358-11ea-95da-452245803204.gif) ![GIF of Normal Web App](https://drive.google.com/open?id=1ltH-eof4RJ6YDXjtEVq33E_2lI-F6AE8)

