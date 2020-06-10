
import os
import parameters
import stock_prediction
import test
import train


def main():
    while True:
        print("LSTM stock prediction")
        print("Please, enter the index you would like to predict")
        ind = input("Index: ").upper()
        ticker = ind
        if ind == 'AMD':
            print('Enter 1 if you want to train on prepared data')
            print('Enter 2 if you want to train on raw data')
            val = input('Val: ').upper()
            if val == '1':
                TRAIN_RAW = True
            else:
                TRAIN_RAW = False

        print("Enter the day which you want to predict after today (e.g. 1 means the next day)")
        day = input("Day: ").upper()
        LOOKUP_STEP = day
        print("Starting training the model")
        os.system("python train.py")
        print("Training is finished. Enter 'test' to start testing")
        t = input("")
        if t == 'test':
            os.system("python test.py")

if __name__ == '__main__':
    main()
