import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)



def data_pre():
    train = pd.read_csv('input/jinnan_round1_train_20181227.csv', encoding='gb2312')
    test = pd.read_csv('input/jinnan_round1_testA_20181227.csv', encoding='gb2312')

    # 部分异常值处理，其他的在文件中修改
    train.iloc[314,9] = '0:00:00'
    train.iloc[386,5] = '20:30:00'
    train.iloc[586,5] = '14:00:00'
    train.iloc[700,26] = '13:00:00'
    train.iloc[998,16] = '12:00:00'
    train.iloc[1140,5] = '20:30:00'
    train.iloc[1320,11] = '21:30:00'

    train.iloc[538,9] = '6:30:00'
    train.iloc[1079,11] = '0:30:00'

    train.iloc[141, 20] = '6:00-6:30'
    train.iloc[799, 20] = '6:00-6:30'
    train.iloc[786, 20] = '6:00-6:30'

    train.drop(1304,inplace=True)
    train['A25'] = train['A25'].astype(int)


    test.iloc[86,5] = '22:00:00'

    col = train.columns.tolist()[1:-1]
    del_col = []
    for each in col:
        if (train[each].count()==train.shape[0] and train[each].nunique()==1) or \
                (test[each].count() == test.shape[0] and test[each].nunique() == 1):
            del_col.append(each)

    data = pd.concat([train,test])
    data = data[[col for col in data.columns if col not in del_col]]


    # A5 为基线，设为0：00：00
    durations = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']
    time = ['A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26','B5', 'B7']


    def time_format(a5, col):
        if pd.isnull(col):
            return col

        a5 = datetime.datetime.strptime(a5, '%H:%M:%S')
        col = datetime.datetime.strptime(col, '%H:%M:%S')
        gap = col - a5
        return gap.seconds
    for each in time:
        data[each] = data.apply(lambda x: time_format(x['A5'], x[each]), axis=1)

    def durations_last(col):
        if pd.isnull(col):
            return col
        gap = datetime.datetime.strptime(col.split('-')[1], '%H:%M') - datetime.datetime.strptime(col.split('-')[0], '%H:%M')

        return gap.seconds

    def durations_format(a5, col):
        if pd.isnull(col):
            return col
        a5 = datetime.datetime.strptime(a5, '%H:%M:%S')
        gap = datetime.datetime.strptime(col.split('-')[0], '%H:%M') - a5
        return gap.seconds


    for each in durations:
        data[each+'_lasttime'] = data[each].apply(lambda x: durations_last(x))
        data[each] = data.apply(lambda x: durations_format(x['A5'], x[each]), axis=1)

    data.drop('A5',axis=1,inplace=True)

    return data.loc[:train.shape[0]], data.iloc[train.shape[0]:]