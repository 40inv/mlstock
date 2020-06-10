import csv
import os
from datetime import datetime
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np


def predict_for_certain_company(filename):
    #to get min max
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'data//'+filename+'.csv'))
    closePriceTraining = []
    with open(path, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            closePriceTraining.append(float(row[4]))
    
    minValueOfClose = min(closePriceTraining)
    maxValueOfClose = max(closePriceTraining)

    #load data for checking
    pathC = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'dataComparision//'+filename+'.csv'))
    openPriceCheck = []
    closePriceCheck = []
    datesCheck = []
    with open(pathC, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            date = datetime.strptime(row[0], '%Y-%m-%d').date()
            dtmp = (datetime.combine(date, datetime.min.time()) - datetime(1970, 1, 1)).days
            #openPriceCheck.append([dtmp,float(row[1])])
            openPriceCheck.append([float(row[1])])
            closePriceCheck.append(float(row[4]))
            datesCheck.append(datetime.strptime(row[0], '%Y-%m-%d').date())

    closePriceCheck = (np.array(closePriceCheck)-minValueOfClose)/(maxValueOfClose-minValueOfClose) # scale from 0 to 1
    i = 0
    while i < len(openPriceCheck):
        openPriceCheck[i][0] = (openPriceCheck[i][0]-minValueOfClose)/(maxValueOfClose-minValueOfClose)
        #openPriceCheck[i][1] = (openPriceCheck[i][1]-minValueOfClose)/(maxValueOfClose-minValueOfClose)
        i = i + 1

    #dates for prediction
    datesPredict = []
    closePriceActual = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'dataPrediction//'+filename+'.csv'))
    with open(path, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            datesPredict.append(datetime.strptime(row[0], '%Y-%m-%d').date())
            closePriceActual.append(float(row[4]))

    pathSVR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'trainedModels//withScaling//'+filename+'svr.joblib'))
    svr_rbf = load(pathSVR)

    closePriceActual = (np.array(closePriceActual)-minValueOfClose)/(maxValueOfClose-minValueOfClose) # scale from 0 to 1
    i = 0
    dtmp = (datetime.combine(datesPredict[0], datetime.min.time()) - datetime(1970, 1, 1)).days

    #openPriceFromPreviousClose = [dtmp,closePriceCheck[len(closePriceCheck)-1]]
    openPriceFromPreviousClose = [closePriceCheck[len(closePriceCheck)-1]]
    predictedPrices = [svr_rbf.predict([openPriceFromPreviousClose])]
    naivePrediction = [closePriceCheck[len(closePriceCheck)-7]* (maxValueOfClose - minValueOfClose) + minValueOfClose]

    i = 1
    while i < len(datesPredict):
        dtmp = (datetime.combine(datesPredict[i], datetime.min.time()) - datetime(1970, 1, 1)).days
        #openPriceFromPreviousClose = [dtmp, predictedPrices[i-1]]
        #openPriceFromPreviousClose = [predictedPrices[i-1]]
        naivePrediction.append(closePriceCheck[len(closePriceCheck)-7+i]* (maxValueOfClose - minValueOfClose) + minValueOfClose)
        openPriceFromPreviousClose = predictedPrices[i-1]
        predictedPrices.append(svr_rbf.predict([openPriceFromPreviousClose]))
        i = i + 1

    i = 0
    while i < len(closePriceActual):
        print("error:")
        error = ((predictedPrices[i]-closePriceActual[i])/(closePriceActual[i])) * 100
        print(error)
        i += 1
        
    predictedPrices = np.array(predictedPrices) * (maxValueOfClose - minValueOfClose) + minValueOfClose
    openPredictPricesBestSVR = svr_rbf.predict(openPriceCheck)
    openPredictPricesBestSVR = openPredictPricesBestSVR * (maxValueOfClose - minValueOfClose) + minValueOfClose
    closePriceCheck = closePriceCheck * (maxValueOfClose - minValueOfClose) + minValueOfClose
    closePriceActual = closePriceActual * (maxValueOfClose - minValueOfClose) + minValueOfClose

    plt.plot(datesPredict, predictedPrices, color='green', label='SVR rbf prediction')
    plt.plot(datesPredict, naivePrediction, color='blue', label='Naive prediction')
    plt.plot(datesCheck, openPredictPricesBestSVR, color='red', label='SVR rbf Trained Predicted')
    plt.scatter(datesCheck, closePriceCheck, color='red', label='Actual data for prediction ')
    plt.scatter(datesPredict, closePriceActual, color='black', label='Data')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Regression')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    filename = 'BABA' # without csv please
    predict_for_certain_company(filename)