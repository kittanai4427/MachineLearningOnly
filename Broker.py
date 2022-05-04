from tensorflow import keras
import pandas as pd
from joblib import dump, load
import pickle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import load_model
from sklearn import preprocessing


def clean(df):
    # categorical_featureX = df.select_dtypes(include=['object']).columns
    # df = pd.get_dummies(df, columns=categorical_featureX)

    standard_scaler = getScaler()
    X = pd.DataFrame(standard_scaler.transform(df.values),index = df.index, columns=df.columns)

    # data_Corr = X.corr()
    # lower = pd.DataFrame(np.tril(data_Corr,-1),columns=data_Corr.columns)
    to_drop = ['Smoking_No', 'AlcoholDrinking_No', 'Stroke_No', 'DiffWalking_No', 'Sex_Female', 'Diabetic_No', 'PhysicalActivity_No', 'Asthma_No', 'KidneyDisease_No', 'SkinCancer_No']
    # to_drop = [column for column in lower if any(abs(lower[column]) > 0.86)]
    X.drop(to_drop,inplace=True,axis = 1)
    return X

def getScaler():

    scaler = load('std_scaler.bin')
    return scaler

def getKNN():

    model = pickle.load(open('modelKNN.pkl', "rb"))
    return model

def getNN():

    model = load_model('NN.model')
    return model


def fillData(df=None):
    columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime','Smoking_No', 'Smoking_Yes','AlcoholDrinking_No', 'AlcoholDrinking_Yes', 'Stroke_No','Stroke_Yes',
               'DiffWalking_No','DiffWalking_Yes', 'Sex_Female','Sex_Male',
               'AgeCategory_18-24', 'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44',
               'AgeCategory_45-49', 'AgeCategory_50-54', 'AgeCategory_55-59',
               'AgeCategory_60-64', 'AgeCategory_65-69', 'AgeCategory_70-74', 'AgeCategory_75-79',
               'AgeCategory_80 or older', 'Race_American Indian/Alaskan Native', 'Race_Asian', 'Race_Black',
               'Race_Hispanic', 'Race_Other', 'Race_White', 'Diabetic_No, borderline diabetes', 'Diabetic_No','Diabetic_Yes',
               'Diabetic_Yes (during pregnancy)', 'PhysicalActivity_No','PhysicalActivity_Yes', 'GenHealth_Excellent', 'GenHealth_Fair',
               'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good','Asthma_No' ,'Asthma_Yes','KidneyDisease_No', 'KidneyDisease_Yes',
               'SkinCancer_No','SkinCancer_Yes']
    l = []
    subL = []
    for i in range(len(columns)):
        subL.append(0)
    l.append(subL)
    X = pd.DataFrame(data=l, columns=columns)
    print(X.info())
    X["BMI"] = X["BMI"].replace(0, df["BMI"])
    X["PhysicalHealth"] = X["PhysicalHealth"].replace(0, df["PhysicalHealth"])
    X["MentalHealth"] = X["MentalHealth"].replace(0, df["MentalHealth"])
    X["SleepTime"] = X["SleepTime"].replace(0, df["SleepTime"])
    if df["Smoking"] == "Yes":
        X["Smoking_Yes"] = X["Smoking_Yes"].replace(0, 1)
    if df["AlcoholDrinking"] == "Yes":
        X["AlcoholDrinking_Yes"] = X["AlcoholDrinking_Yes"].replace(0, 1)
    if df["Stroke"] == "Yes":
        X["Stroke_Yes"] = X["Stroke_Yes"].replace(0, 1)
    if df["Stroke"] == "Yes":
        X["Stroke_Yes"] = X["Stroke_Yes"].replace(0, 1)
    if df["DiffWalking"] == "Yes":
        X["DiffWalking_Yes"] = X["DiffWalking_Yes"].replace(0, 1)
    if df["Sex"] == "Male":
        X["Sex_Male"] = X["Sex_Male"].replace(0, 1)
    if df["AgeCategory"] == "18-24":
        X["AgeCategory_18-24"] = X["AgeCategory_18-24"].replace(0, 1)
    elif df["AgeCategory"] == "25-29":
        X["AgeCategory_25-29"] = X["AgeCategory_25-29"].replace(0, 1)
    elif df["AgeCategory"] == "30-34":
        X["AgeCategory_30-34"] = X["AgeCategory_30-34"].replace(0, 1)
    elif df["AgeCategory"] == "35-39":
        X["AgeCategory_35-39"] = X["AgeCategory_35-39"].replace(0, 1)
    elif df["AgeCategory"] == "40-44":
        X["AgeCategory_40-44"] = X["AgeCategory_40-44"].replace(0, 1)
    elif df["AgeCategory"] == "45-49":
        X["AgeCategory_45-49"] = X["AgeCategory_45-49"].replace(0, 1)
    elif df["AgeCategory"] == "50-54":
        X["AgeCategory_50-54"] = X["AgeCategory_50-54"].replace(0, 1)
    elif df["AgeCategory"] == "55-59":
        X["AgeCategory_55-59"] = X["AgeCategory_55-59"].replace(0, 1)
    elif df["AgeCategory"] == "60-64":
        X["AgeCategory_60-64"] = X["AgeCategory_60-64"].replace(0, 1)
    elif df["AgeCategory"] == "65-69":
        X["AgeCategory_65-69"] = X["AgeCategory_65-69"].replace(0, 1)
    elif df["AgeCategory"] == "70-74":
        X["AgeCategory_70-74"] = X["AgeCategory_70-74"].replace(0, 1)
    elif df["AgeCategory"] == "75-79":
        X["AgeCategory_75-79"] = X["AgeCategory_75-79"].replace(0, 1)
    else:
        X["AgeCategory_80 or older"] = X["AgeCategory_80 or older"].replace(0, 1)

    if df["Race"] == "American Indian/Alaskan Native":
        X["Race_American Indian/Alaskan Native"] = X["Race_American Indian/Alaskan Native"].replace(0, 1)
    elif df["Race"] == "Asian":
        X["Race_Asian"] = X["Race_Asian"].replace(0, 1)
    elif df["Race"] == "Black":
        X["Race_Black"] = X["Race_Black"].replace(0, 1)
    elif df["Race"] == "White":
        X["Race_White"] = X["Race_White"].replace(0, 1)
    elif df["Race"] == "Hispanic":
        X["Race_Hispanic"] = X["Race_Hispanic"].replace(0, 1)
    else:
        X["Race_Other"] = X["Race_Other"].replace(0, 1)

    if df["Diabetic"] == "Yes":
        X["Diabetic_Yes"] = X["Diabetic_Yes"].replace(0, 1)
    elif df["Diabetic"] == "Yes2":
        X["Diabetic_Yes (during pregnancy)"] = X["Diabetic_Yes (during pregnancy)"].replace(0, 1)
    elif df["Diabetic"] == "No2":
        X["Diabetic_No, borderline diabetes"] = X["Diabetic_No, borderline diabetes"].replace(0, 1)

    if df["PhysicalActivity"] == "Yes":
        X["PhysicalActivity_Yes"] = X["PhysicalActivity_Yes"].replace(0, 1)

    if df["GenHealth"] == 0:
        X["GenHealth_Poor"] = X["GenHealth_Poor"].replace(0, 1)
    elif df["GenHealth"] == 1:
        X["GenHealth_Fair"] = X["GenHealth_Fair"].replace(0, 1)
    elif df["GenHealth"] == 2:
        X["GenHealth_Good"] = X["GenHealth_Good"].replace(0, 1)
    elif df["GenHealth"] == 3:
        X["GenHealth_Very good"] = X["GenHealth_Very good"].replace(0, 1)
    elif df["GenHealth"] == 4:
        X["GenHealth_Excellent"] = X["GenHealth_Excellent"].replace(0, 1)

    if df["Asthma"] == "Yes":
        X["Asthma_Yes"] = X["Asthma_Yes"].replace(0, 1)
    if df["KidneyDisease"] == "Yes":
        X["KidneyDisease_Yes"] = X["KidneyDisease_Yes"].replace(0, 1)
    if df["SkinCancer"] == "Yes":
        X["SkinCancer_Yes"] = X["SkinCancer_Yes"].replace(0, 1)

    return X

def predict(df=None):
    model = getNN()
    pred = model.predict(df)
    return pred

if __name__ == '__main__':
    #
    # df = {"BMI": 21.68, "Smoking": "No", "AlcoholDrinking": "No",
    #         "Stroke": "No", "PhysicalHealth": 15.0, "MentalHealth": 0.0,
    #         "DiffWalking": "No", "Sex": "Female", "AgeCategory": "70-74", "Race": 'White',
    #         "Diabetic": "No", "PhysicalActivity": "Yes", "GenHealth": 1,
    #         "SleepTime": 4.0, "Asthma": "Yes", "KidneyDisease": "No", "SkinCancer": "Yes"}
    # X = fillData(df)
    # X = clean(X)
    #
    # model = getKNN()
    #
    df = pd.read_csv("heart_2020_cleaned.csv")
    df.head()
    Y = pd.DataFrame(df['HeartDisease'])
    df = df.drop(df[['HeartDisease']], axis=1)

    # Y = pd.DataFrame(df['HeartDisease'])
    categorical_featureX = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_featureX)
    model = getKNN()
    X = clean(df)
    # print(X)

    lb = LabelBinarizer()
    labels = lb.fit_transform(Y)
    labels = to_categorical(labels)
    Y = np.array(labels)
    #
    #
    #
    realNN = np.asarray(Y[:11])
    realNN = pd.DataFrame(realNN)
    y_realNN = realNN.idxmax(axis=1)
    print(y_realNN)
    #
    result = model.predict(X.iloc[:11])

    predNN = pd.DataFrame(result)
    y_predNN = predNN.idxmax(axis=1)
    print(y_predNN)

    correct = 0
    for i in range(len(y_predNN)):
        if y_predNN[i] == y_realNN[i]:
            correct+=1
    print('Correct',correct)

