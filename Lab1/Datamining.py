import pandas as pd
import numpy as np
import xlsxwriter
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
workbook = xlsxwriter.Workbook('result.xlsx')
dataset = pd.read_csv('winequality-red.csv')

print(dataset.head())
select_value = int(input("Please select a numbers: (1) summary \n (2)  replace \n (3) discretize \n (4)normalize \n"))
#Option 1:
if select_value == 1: 
    with open("D:/trang download/Datamining/Datamining/log_text.txt","w") as f:
        a = dataset["fixed acidity"].count()
        f.write("Number of instances: " + repr(a) + "\n")
        f.write("Number of attributes: " + repr(dataset.shape[1]) + "\n")
        if dataset["fixed acidity"].dtype == "int64":
            f.write("Name: fixed acidity, type: Numeric\n")
        else:
            f.write("Name: fixed acidity, type: Norminal\n")

        if dataset["volatile acidity"].dtype == "int64":
            f.write("Name: volatile acidity, type: Numeric\n")
        else:
            f.write("Name: volatile acidity, type: Norminal\n")

        if dataset["citric acid"].dtype == "int64":
            f.write("Name: citric acid, type: Numeric\n")
        else:
            f.write("Name: citric acid, type: Norminal\n")

        if dataset["residual sugar"].dtype == "int64":
            f.write("Name: residual sugar, type: Numeric\n")
        else:
            f.write("Name: residual sugar, type: Norminal\n")

        if dataset["chlorides"].dtype == "int64":
            f.write("Name: chlorides, type: Numeric\n")
        else:
            f.write("Name: chlorides, type: Norminal\n")

        if dataset["free sulfur dioxide"].dtype == "int64":
            f.write("Name: free sulfur dioxide, type: Numeric\n")
        else:
            f.write("Name: free sulfur dioxide, type: Norminal\n")

        if dataset["total sulfur dioxide"].dtype == "int64":
            f.write("Name: total sulfur dioxide, type: Numeric\n")
        else:
            f.write("Name: total sulfur dioxide, type: Norminal\n")
        if dataset["density"].dtype == "int64":
            f.write("Name: density, type: Numeric\n")
        else:
            f.write("Name: density, type: Norminal\n")
        if dataset["pH"].dtype == "int64":
            f.write("Name: pH, type: Numeric\n")
        else:
            f.write("Name: pH, type: Norminal\n")
        if dataset["sulphates"].dtype == "int64":
            f.write("Name: sulphates, type: Numeric\n")
        else:
            f.write("Name: sulphates, type: Norminal\n")
        if dataset["alcohol"].dtype == "int64":
            f.write("Name: alcohol, type: Numeric\n")
        else:
            f.write("Name: alcohol, type: Norminal\n")
        if dataset["quality"].dtype == "int64":
            f.write("Name: quality, type: Numeric\n")
        else:
            f.write("Name: quality, type: Norminal\n")
#Option2:
elif select_value  == 2:
    missing_values = "?"
    for values in dataset.iterrows():
        if missing_values == "?":
            missing_values = input()
#Option3:
elif select_value == 3:
        binning = int(input("(1)equal frequency binning \n (2)equal width binning \n"))
        Names = int(input("Please choose 1 attribute for binning: \n (1)fixed acidity \n (2)volatile acidity \n (3)citric acid \n (4)residual sugar \n (5)chlorides \n (6)free sulfur dioxide \n (7)total sulfur dioxide \n (8)density \n (9)pH \n (10)sulphates \n (11)alcohol \n (12)quality \n"))
        bin = int(input("Please choose the number of bins: "))
        if binning ==1 and Names==1:
          df = pd.qcut(dataset['fixed acidity'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['fixed acidity'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==2:
          df = pd.qcut(dataset['volatile acidity'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['volatile acidity'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==3:
          df = pd.qcut(dataset['citric acid'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['citric acid'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==4:
          df = pd.qcut(dataset['residual sugar'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['residual sugar'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==5:
          df = pd.qcut(dataset['chlorides'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['chlorides'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==6:
          df = pd.qcut(dataset['free sulfur dioxide'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['free sulfur dioxide'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==7:
          df = pd.qcut(dataset['total sulfur dioxide'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['total sulfur dioxide'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==8:
          df = pd.qcut(dataset['density'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['density'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==9:
          df = pd.qcut(dataset['pH'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['pH'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==10:
          df = pd.qcut(dataset['sulphates'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['sulphates'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==11:
          df = pd.qcut(dataset['alcohol'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['alcohol'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==1 and Names ==12:
          df = pd.qcut(dataset['quality'],q=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.qcut(dataset['quality'],q=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==1:
          df = pd.cut(dataset['fixed acidity'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['fixed acidity'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==2:
          df = pd.cut(dataset['volatile acidity'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['volatile acidity'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==3:
          df = pd.cut(dataset['citric acid'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['citric acid'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==4:
          df = pd.cut(dataset['residual sugar'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['residual sugar'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==5:
          df = pd.cut(dataset['chlorides'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['chlorides'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==6:
          df = pd.cut(dataset['free sulfur dioxide'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['free sulfur dioxide'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==7:
          df = pd.cut(dataset['total sulfur dioxide'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['total sulfur dioxide'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==8:
          df = pd.cut(dataset['density'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['density'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==9:
          df = pd.cut(dataset['pH'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['pH'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==10:
          df = pd.cut(dataset['sulphates'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['sulphates'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==11:
          df = pd.cut(dataset['alcohol'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['alcohol'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")
        if binning ==2 and Names ==12:
          df = pd.cut(dataset['quality'],bins=bin)
          df1 = df.value_counts()
          group_bin =[]
          group_bin.append(pd.cut(dataset['quality'],bins=bin))
          dataset=pd.DataFrame(group_bin).T
          dataset.to_excel(excel_writer ="D:/trang download/Datamining/Datamining/result.xlsx")
          df1.to_excel(excel_writer="D:/trang download/Datamining/Datamining/log_text.xlsx")

#Option 4:
elif select_value ==4:
    solution = int(input("Please choose normalization solution: \n (1)Min-Max \n (2)Z-score \n" ))
    if solution == 1:
        TypeOfData = int(input("Please choose type of input: \n (1)Có dữ liệu thiếu cho cả dữ liệu nominal và numeric, với số lượng mẫu là 20 \n (2)Dữ liệu đầy đủ với số lượng mẫu là 20. \n (3)Dữ liệu đầy đủ với số lượng mẫu lớn hơn 100. \n" ))
        if TypeOfData ==1:
            df= pd.read_csv('Type 1.csv')
            data = df.values

            scaler = MinMaxScaler()
            scaler.fit(data)

            scaled_data = scaler.transform(data)
            cf = pd.DataFrame(scaled_data)
            cf.to_excel(excel_writer="D:/trang download/Datamining/Datamining/result.xlsx")
        if TypeOfData ==2:
            df= pd.read_csv('Type 2.csv')
            data = df.values

            scaler = MinMaxScaler()
            scaler.fit(data)

            scaled_data = scaler.transform(data)
            cf = pd.DataFrame(scaled_data)
            cf.to_excel(excel_writer="D:/trang download/Datamining/Datamining/result.xlsx")
        if TypeOfData ==3:
            df= pd.read_csv('winequality-red.csv')
            data = df.values

            scaler = MinMaxScaler()
            scaler.fit(data)

            scaled_data = scaler.transform(data)
            cf = pd.DataFrame(scaled_data)
            cf.to_excel(excel_writer="D:/trang download/Datamining/Datamining/result.xlsx")

    if solution == 2:
        TypeOfData = int(input("Please choose type of input: \n (1)Có dữ liệu thiếu cho cả dữ liệu nominal và numeric, với số lượng mẫu là 20 \n (2)Dữ liệu đầy đủ với số lượng mẫu là 20. \n (3)Dữ liệu đầy đủ với số lượng mẫu lớn hơn 100. \n"))
        if TypeOfData ==1:
            df=pd.read_csv('Type 1.csv')
            result =stats.zscore(df)
            cf = pd.DataFrame(result)
            cf.to_excel(excel_writer="D:/trang download/Datamining/Datamining/result.xlsx")
        if TypeOfData ==2:
            df=pd.read_csv('Type 2.csv')
            result =stats.zscore(df)
            cf = pd.DataFrame(result)
            cf.to_excel(excel_writer="D:/trang download/Datamining/Datamining/result.xlsx")
        if TypeOfData ==3:
            df=pd.read_csv('winequality-red.csv')
            result =stats.zscore(df)
            cf = pd.DataFrame(result)
            cf.to_excel(excel_writer="D:/trang download/Datamining/Datamining/result.xlsx")
