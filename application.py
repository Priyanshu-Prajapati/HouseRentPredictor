from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__) 

model = pickle.load(open("LinearRegressionHouseModel.pkl", 'rb'))

house = pd.read_csv('Cleaned_house_details.csv')

@app.route('/')
def index():
    areaType = house['Area Type'].unique()
    city = house['City'].unique()
    furnishingStatus = house['Furnishing Status'].unique()
    return render_template('testindex.html', AreaType=areaType, City=city, FurnishingStatus=furnishingStatus)

@app.route('/predict', methods=['POST'])
def predict():

    AreaType = request.form.get('AreaType')
    City = request.form.get('City')
    FurnishingStatus = request.form.get('FurnishingStatus')
    BHK = int(request.form.get('BHK'))
    HouseSize = int(request.form.get('HouseSize'))
    Floor = int(request.form.get('Floor'))
    Bathroom = int(request.form.get('Bathroom'))

    prediction = model.predict(pd.DataFrame([[BHK, HouseSize, Floor, AreaType, City, FurnishingStatus, Bathroom]], columns = ['BHK', 'Size', 'Floor', 'Area Type', 'City', 'Furnishing Status', 'Bathroom']))
   
    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug = True)