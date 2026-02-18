#pycaret library is used to implement explainable ai with fewer lines of code

import pandas as pd
from pycaret.datasets import get_data
from pycaret.classification import *

data = get_data('diabetes')

pd.set_option('display.width', 1000) #This line show all columns in same one line
pd.set_option('display.max_columns',None) #This line do not show .... for columns instead all names of columns

print(data.head())

new_data = data.rename(columns={'Number of times pregnant':'Pregnancies','Plasma glucose concentration a 2 hours in an oral glucose tolerance test':'Glucose','Diastolic blood pressure (mm Hg)':'BloodPressure','Triceps skin fold thickness (mm)':'SkinThickness','2-Hour serum insulin (mu U/ml)':'Insulin','Body mass index (weight in kg/(height in m)^2)':'BMI','Diabetes pedigree function':'Pedigree','Age (years)':'Age','Class variable':'Outcome'})

print(new_data.head())

s = setup(new_data,target='Outcome',session_id=43) #setup function do train_test splitting,cleaning,target finding and finding best model .

best_model = compare_models()
pull()
get_metrics()

print(best_model)

print("Creating model  Random Forest,as lrp is good but random forest is complex so it is selected")
model = create_model('rf')

evaluate_model(model)

print('Generating summary plot (which show which column is important for the whole dataset)')
interpret_model(model,plot='summary')

print('Generating reason plot,it tells the reason for a specific patient')
interpret_model(model,plot='reason')

plot_model(model,plot='feature')  #which features matter most for all patients.

single_patient_row = new_data.iloc[[0]]  #Selecting one patient (first row) for explanation

prediction = predict_model(model, data=single_patient_row) #shows only yes or no (probability)

print(prediction)

interpret_model(model,plot='reason',observation=0) #Why the model gave a specific prediction for ONE particular person (row).

predict_model(model, data=new_data) #The model predicts YES with 87% confidence.



# ---------------------- NATURAL LANGUAGE EXPLANATION PART ----------------------

#Extracting prediction label and probability
predicted_class = prediction['prediction_label'].values[0]
prediction_score = prediction['prediction_score'].values[0]

print("\n----------- NATURAL EXPLANATION -----------")

#Converting model output into simple human readable explanation
if predicted_class == 1:
    print(f"The model predicts that the patient is likely Diabetic with {round(prediction_score*100,2)}% confidence.")
else:
    print(f"The model predicts that the patient is NOT Diabetic with {round(prediction_score*100,2)}% confidence.")

print("This decision is mainly influenced by important features such as Glucose, BMI, Age and other health indicators shown in the explanation plot above.")

print("--------------------------------------------")
