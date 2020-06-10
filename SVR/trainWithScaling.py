import csv
import os
from datetime import datetime
from sklearn.svm import SVR
from joblib import dump, load
from sklearn import preprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



def train_for_certain_company(filename, maxCountC, maxCountGamma):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'data//'+filename+'.csv'))

    # load data for train
    openPriceTraining = []
    closePriceTraining = []
    datesTraining = []
    with open(path, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            date = datetime.strptime(row[0], '%Y-%m-%d').date()
            dtmp = (datetime.combine(date, datetime.min.time()) - datetime(1970, 1, 1)).days
            openPriceTraining.append([dtmp,float(row[1])])
            #openPriceTraining.append([float(row[1])])
            closePriceTraining.append(float(row[4]))
            datesTraining.append(datetime.strptime(row[0], '%Y-%m-%d').date())
    
    minValueOfClose = min(closePriceTraining)
    maxValueOfClose = max(closePriceTraining)
    closePriceTraining = (np.array(closePriceTraining)-minValueOfClose)/(maxValueOfClose-minValueOfClose) # scale from 0 to 1
    i = 0
    while i < len(openPriceTraining):
        #openPriceTraining[i][0] = (openPriceTraining[i][0]-minValueOfClose)/(maxValueOfClose-minValueOfClose)
        openPriceTraining[i][1] = (openPriceTraining[i][1]-minValueOfClose)/(maxValueOfClose-minValueOfClose)
        i = i + 1
    
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
            openPriceCheck.append([dtmp,float(row[1])])
            #openPriceCheck.append([float(row[1])])
            closePriceCheck.append(float(row[4]))
            datesCheck.append(datetime.strptime(row[0], '%Y-%m-%d').date())
    
    closePriceCheck = (np.array(closePriceCheck)-minValueOfClose)/(maxValueOfClose-minValueOfClose) # scale from 0 to 1
    i = 0
    while i < len(openPriceCheck):
        openPriceCheck[i][1] = (openPriceCheck[i][1]-minValueOfClose)/(maxValueOfClose-minValueOfClose)
        #openPriceCheck[i][0] = (openPriceCheck[i][0]-minValueOfClose)/(maxValueOfClose-minValueOfClose)
        i = i + 1

    #trainig
    svr_rbf = SVR(kernel='rbf', C=1, gamma=1e-9, epsilon=.1) 
    svr_rbf.fit(openPriceTraining, closePriceTraining)
    predictedPrices = svr_rbf.predict(openPriceCheck)
    i = 0
    symForMSE = 0
    while i < len(closePriceCheck): 
        symForMSE = symForMSE + (predictedPrices[i] - closePriceCheck[i]) ** 2
        i += 1
    errorPred = symForMSE / len(closePriceCheck)
    bestErrorPred = errorPred

    loopCountGamma = 1 # since one is trained before main loop
    loopCountC = 0

    C = 1
    bestC = 1
    Gamma = 1e-8
    bestGamma= 1e-9
    bestSvr = svr_rbf

    while loopCountC < maxCountC:
        print(loopCountC)
        while loopCountGamma < maxCountGamma:
            print(Gamma, C, errorPred)
            # Create the Gaussian RBF Kernel Support Vector Regression model
            svr_rbf = SVR(kernel='rbf', C=C, gamma=Gamma, epsilon=.1)  
            # Train the SVR models
            svr_rbf.fit(openPriceTraining, closePriceTraining)
            # Predict results
            predictedPrices = svr_rbf.predict(openPriceCheck)
            #Compare thdatesCem with true results using MSE
            symForMSE = 0
            i = 0
            while i < len(closePriceCheck):
                symForMSE = symForMSE + (predictedPrices[i]-closePriceCheck[i])**2
                i += 1
            errorPred = symForMSE / len(closePriceCheck)
            if errorPred < bestErrorPred:
                bestSvr = svr_rbf
                bestErrorPred = errorPred
                bestC = C
                bestGamma = Gamma
            loopCountGamma = loopCountGamma + 1
            Gamma = Gamma * 10

        C = C * 10
        loopCountGamma = 0
        loopCountC = loopCountC + 1
        Gamma = 1e-9

    print("Gamma: ", bestGamma, " C: ", bestC, " Error: ", bestErrorPred)
    # save trained model
    Path("./trainedModels/withScaling").mkdir(parents=True, exist_ok=True)
    pathSVR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'trainedModels//withScaling//'+filename))
    dump(bestSvr, pathSVR+'svr.joblib')
    openPricesBestSVR = bestSvr.predict(openPriceTraining)
    openPricesBestSVR = openPricesBestSVR * (maxValueOfClose - minValueOfClose) + minValueOfClose
    openPredictPricesBestSVR = bestSvr.predict(openPriceCheck)
    openPredictPricesBestSVR = openPredictPricesBestSVR * (maxValueOfClose - minValueOfClose) + minValueOfClose
    closePriceTraining = closePriceTraining * (maxValueOfClose - minValueOfClose) + minValueOfClose
    closePriceCheck = closePriceCheck * (maxValueOfClose - minValueOfClose) + minValueOfClose

    plt.plot(datesTraining, openPricesBestSVR, color='red', label='SVR rbf Trained')
    plt.plot(datesCheck,openPredictPricesBestSVR, color='green', label='SVR rbf Trained Prdicted')
    plt.scatter(datesTraining, closePriceTraining, color='black', label='Data')
    plt.scatter(datesCheck, closePriceCheck, color='blue', label='Actual data for prediction ')

    plt.xlabel('Open')
    plt.ylabel('Close')
    plt.title('Regression')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # BABA for fast check
    filename = 'BABA' # without csv please
    train_for_certain_company(filename,6, 6)