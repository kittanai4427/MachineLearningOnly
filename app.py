import pandas as pd
from flask import Flask,render_template,request
import Broker as bk
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sendData')
def signupForm():

    BMI = request.args.get('BMI')
    Smoking = request.args.get('Smoking')
    AlcoholDrinking = request.args.get('AlcoholDrinking')
    Stroke = request.args.get('Stroke')
    PhysicalHealth = request.args.get('PhysicalHealth')
    MentalHealth = request.args.get('MentalHealth')
    DiffWalking = request.args.get('DiffWalking')
    Sex = request.args.get('Sex')
    AgeCategory = request.args.get('AgeCategory')
    Race = request.args.get('Race')
    Diabetic = request.args.get('Diabetic')
    PhysicalActivity = request.args.get('PhysicalActivity')
    GenHealth = request.args.get('GenHealth')
    SleepTime = request.args.get('SleepTime')
    Asthma = request.args.get('Asthma')
    KidneyDisease = request.args.get('KidneyDisease')
    SkinCancer = request.args.get('SkinCancer')

    data = {"BMI": 34.3, "Smoking": "Yes", "AlcoholDrinking": "No",
          "Stroke": "No", "PhysicalHealth": 30.0, "MentalHealth": 0.0,
          "DiffWalking": "Yes", "Sex": "Male", "AgeCategory": "25-29", "Race": 'White',
          "Diabetic": "Yes", "PhysicalActivity": "No", "GenHealth": 0,
          "SleepTime": 15.0, "Asthma": "Yes", "KidneyDisease": "No", "SkinCancer": "No"}

    df = bk.fillData(data)

    df = bk.clean(df)
    result = bk.predict(df)
    predNN = pd.DataFrame(result)
    y_predNN = predNN.idxmax(axis=1)

    if y_predNN[0] == 0:
        out = 'คุณไม่เป็นโรคหัวใจ'
    else:
        out = 'คุณเป็นโรคหัวใจ'
    return render_template('thankyou.html',data=out)

if __name__ == '__main__':
    app.run(debug=True)