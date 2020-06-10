import csv
import os
from datetime import date, datetime
import matplotlib.pyplot as plt


def plot_stock_dataset(filename):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'data//'+filename+'.csv'))
    dates = []
    prices = []
    with open(path, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            dates.append(datetime.strptime(row[0], '%Y-%m-%d').date())
            prices.append(float(row[1]))  # Convert to float for more precision

    plt.plot(dates, prices, color='black', label='Price model')  # plotting the line made by the RBF kernel
    plt.xlabel('Date')  # Setting the x-axis
    plt.ylabel('Price')  # Setting the y-axis
    plt.title('Price plot')  # Setting title
    plt.legend()  # Add legend
    plt.show()  # To display result on screen


if __name__ == "__main__":
    filename = 'SPY'
    plot_stock_dataset(filename)
