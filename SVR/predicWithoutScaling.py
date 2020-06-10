import csv
import os
from datetime import datetime
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np

def predict_for_certain_company(filename):

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

    datesPredict = []
    closePriceActual = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'dataPrediction//'+filename+'.csv'))
    with open(path, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            datesPredict.append(datetime.strptime(row[0], '%Y-%m-%d').date())
            closePriceActual.append(float(row[4]))

    pathSVR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'trainedModels//withoutScaling//'+filename+'svr.joblib'))
    svr_rbf = load(pathSVR)

    dtmp = (datetime.combine(datesPredict[0], datetime.min.time()) - datetime(1970, 1, 1)).days

    #openPriceFromPreviousClose = [dtmp,closePriceCheck[len(closePriceCheck)-1]]
    naivePrediction = [closePriceCheck[len(closePriceCheck)-7]]
    openPriceFromPreviousClose = [closePriceCheck[len(closePriceCheck)-1]]
    predictedPrices = [svr_rbf.predict([openPriceFromPreviousClose])]
    i = 1
    while i < len(datesPredict):
        dtmp = (datetime.combine(datesPredict[i], datetime.min.time()) - datetime(1970, 1, 1)).days
        #openPriceFromPreviousClose = [dtmp, predictedPrices[i-1]]
        #openPriceFromPreviousClose = [predictedPrices[i-1]]
        openPriceFromPreviousClose = predictedPrices[i-1]
        naivePrediction.append(closePriceCheck[len(closePriceCheck)-7+i])
        predictedPrices.append(svr_rbf.predict([openPriceFromPreviousClose]))
        i = i + 1

    i = 0
    while i < len(closePriceActual):
        print("error:")
        error = ((predictedPrices[i]-closePriceActual[i])/(closePriceActual[i])) * 100
        print(error)
        i += 1


    plt.plot(datesPredict, predictedPrices, color='green', label='SVR rbf prediction')
    plt.plot(datesPredict, naivePrediction, color='blue', label='Naive prediction')
    plt.plot(datesCheck, svr_rbf.predict(openPriceCheck), color='red', label='SVR rbf pretrained prediction')
    plt.scatter(datesCheck, closePriceCheck, color='red', label='Actual data for pretrained prediction ')
    plt.scatter(datesPredict, closePriceActual, color='black', label='Data')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Regression')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    filename = 'AAPL' # without csv please
    predict_for_certain_company(filename)