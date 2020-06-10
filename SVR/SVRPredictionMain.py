import plot_data
import prepare_data
import trainWithoutScaling
import predicWithoutScaling
import trainWithScaling
import predicWithScaling

menuActions = ['1. Download datasets',
               '2. Plot data',
               '3. Train SVR without scaling',
               '4. Predict using trained data without scaling',
               '5. Train SVR with scaling',
               '6. Predict using trained data with scaling'
               ]

if __name__ == "__main__":
    while True:
        print("SVR stock prediction")
        print("Text menu")
        for menuItem in menuActions:
            print(menuItem)
        action = input("Action: ").upper()
        if action not in "123456" or len(action) != 1:
            print("I don't know how to do that")
            continue
        if action == '1':
            #print("Data for train")
            #dateStart = input("Input start date yyyy-mm-dd: ").upper()
            #dateEnd = input("Input end date yyyy-mm-dd: ").upper()
            #prepare_data.build_stock_dataset_for_train(dateStart,dateEnd)
            prepare_data.build_stock_dataset_for_train()
            #print("Data for comparision")
            #dateStart = input("Input start date yyyy-mm-dd: ").upper()
            #dateEnd = input("Input end date yyyy-mm-dd: ").upper()
            #prepare_data.build_stock_dataset_for_comparision(dateStart,dateEnd)
            prepare_data.build_stock_dataset_for_comparision()
            #print("Data for prediction")
            #dateStart = input("Input start date yyyy-mm-dd: ").upper()
            #dateEnd = input("Input end date yyyy-mm-dd: ").upper()
            #prepare_data.build_stock_dataset_for_prediction(dateStart,dateEnd)
            prepare_data.build_stock_dataset_for_prediction()
        elif action == '2':
            filename = input("Input filename from data folder without .csv: ").upper()
            plot_data.plot_stock_dataset(filename)
        elif action == '3':
            filename = input("Input filename from data folder without .csv: ").upper()
            maxCountC = input("Input maxCountC for train loop: ").upper()
            maxCountGamma = input("Input maxCountGamma for train: ").upper()
            trainWithoutScaling.train_for_certain_company(filename,int(maxCountC),int(maxCountGamma))
        elif action == '4':
            filename = input("Input filename from data folder without .csv: ").upper()
            predicWithoutScaling.predict_for_certain_company(filename)
        elif action == '5':
            filename = input("Input filename from data folder without .csv: ").upper()
            maxCountC = input("Input maxCountC for train loop: ").upper()
            maxCountGamma = input("Input maxCountGamma for train: ").upper()
            trainWithScaling.train_for_certain_company(filename,int(maxCountC),int(maxCountGamma))
        elif action == '6':
            filename = input("Input filename from data folder without .csv: ").upper()
            predicWithScaling.predict_for_certain_company(filename)

