# MedAI Diagnoser
This web application provides a user-friendly interface for diagnosing various medical conditions using machine learning models. Users can upload medical images or provide relevant information for diagnosis, and the application will provide predictions based on trained models.

## Features:
**Diagnosis for Multiple Conditions**: Includes models for diagnosing conditions such as COVID-19, brain tumors, Alzheimer's disease, diabetes, breast cancer, pneumonia, heart disease, kidney disease, liver disease, and malaria.
**Image Upload**: Users can upload medical images for analysis, which are then processed by the respective image processing pipelines before making predictions.
**Input Form**: For conditions that require input data (such as diabetes and heart disease), users can fill out a form with relevant information for diagnosis.
**Real-time Chatbot**: Users can interact with a chatbot powered by state-of-the-art language models to ask questions and receive medical advice or information.




![home](https://github.com/pratik9409/Multi_cure/assets/67755812/fd7ce100-31a8-4506-8556-d386ed4aba8d)




# Accuracy of Prediction for all Diseases:


| Disease        | Accuracy      | Dataset Link                                                                                             | Models          |
| -------------  |:-------------:| :-------------------------------------------------------------------------------------------------------: | --------------: |
|  Alzheimer     | 87.68%        | [Alzheimer Data](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)         | Deep Learning| 
| Breast Cancer  | 98.25%        | [Breast cancer Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)                 | Deep Learning   |
| Brain Tumor    | 83.82%        | [Brain Tumor Data](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  | Machine Learning|
| Covid-19       | 86.75%        | [Covid-19 Data](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)                   | Deep Learning     |
| Diabetes       | 96.50%         | [Diabetes Data](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)                    | Machnine Learning|
| Heart Disease  | 85.25%        | [Heart Disease Data](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction)              | Machine Learning |
| Kidney Disease | 99.00%        | [Kidney DIsease Data](https://www.kaggle.com/datasets/mansoordaku/ckdisease)                             | Machine Learning |
| Liver Disease  | 77.00%        | [Liver Disease Data](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)                 | Machine Learning |
| Malaria        | 93.70%        | [Malaria Data](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)               | Deep Learning|
| Pneumonia      | 92.23%        | [Pneumonia Data](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)                | Deep Learning|

# Medical Chatbot

![chat_pic](https://github.com/pratik9409/Multi_cure/assets/67755812/75ed7cdf-b08e-4b12-b260-c094dbeaed77)

# Technologies Used:
Backend: Flask framework in Python
Machine Learning Models: TensorFlow, Scikit-learn, OpenCV
Image Processing: PIL, OpenCV
Language Models: GPT-based language models for the chatbot
Generative AI, LLM
# Steps to run this application in your system

#### a. Clone or download the repo.
#### b. Open command prompt in the downloaded folder.
#### c. Create a virtual environment
   `mkvirtualenv environment_name`
#### d. Install all the dependencies:
`pip install -r requirements.txt`
#### e. Run the application
`python app.py`
