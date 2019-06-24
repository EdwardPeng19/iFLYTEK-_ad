import pandas as pd
import numpy as np
import time
DATA = './data/'
MODEL = './model/'



def time_format(data):
    data['nginxtime_format'] = data.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['nginxtime'] / 1000)), axis=1)
    data['nginxtime_date'] = data.apply(lambda x: x['nginxtime_format'].split(' ')[0], axis=1)
    data['nginxtime_time'] = data.apply(lambda x: x['nginxtime_format'].split(' ')[1], axis=1)
    data['nginxtime_hour'] = data.apply(lambda x: x['nginxtime_format'].split(' ')[1].split(':')[0], axis=1)
    return data

def model_input():
    train = pd.read_csv(DATA + 'round1_iflyad_anticheat_traindata.txt', sep='\t', encoding='utf-8')
    test = pd.read_csv(DATA + 'round1_iflyad_anticheat_testdata_feature.txt', sep='\t', encoding='utf-8')
    train = time_format(train) # 2019-06-03 2019-06-09
    test = time_format(test) # 2019-06-10


    train.to_csv(MODEL + 'train.csv', encoding='utf-8', index=False)
    test.to_csv(MODEL + 'test.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    model_input()


