import pandas as pd
import numpy as np
import time
import requests
import json
import datetime
DATA = './data/'
MODEL = './model/'



def time_format(data):
    data['nginxtime_format'] = data.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['nginxtime'] / 1000)), axis=1)
    data['nginxtime_date'] = data.apply(lambda x: x['nginxtime_format'].split(' ')[0], axis=1)
    data['nginxtime_time'] = data.apply(lambda x: x['nginxtime_format'].split(' ')[1], axis=1)
    data['nginxtime_hour'] = data.apply(lambda x: x['nginxtime_format'].split(' ')[1].split(':')[0], axis=1)
    data['nginxtime_week'] = data.apply(lambda x: datetime.datetime.strptime(x['nginxtime_date'], '%Y-%m-%d').weekday(), axis=1)
    return data

def md5_format(data):
    data['adidmd5_0'] = data.apply(lambda x: 0 if x['adidmd5']=='empty' else 1, axis=1)
    data['imeimd5_0'] = data.apply(lambda x: 0 if x['imeimd5'] == 'empty' else 1, axis=1)
    data['idfamd5_0'] = data.apply(lambda x: 0 if x['idfamd5'] == 'empty' else 1, axis=1)
    data['openudidmd5_0'] = data.apply(lambda x: 0 if x['openudidmd5'] == 'empty' else 1, axis=1)
    data['macmd5_0'] = data.apply(lambda x: 0 if x['macmd5'] == 'empty' else 1, axis=1)
    return data
def ip_info(ip):
    url = f'http://ip.360.cn/IPQuery/ipquery?ip={ip}'
    address_detail = json.loads(requests.get(url).text)['data']
    print(address_detail)
    return address_detail

def model_input():
    train = pd.read_csv(DATA + 'round1_iflyad_anticheat_traindata.txt', sep='\t', encoding='utf-8')
    test = pd.read_csv(DATA + 'round1_iflyad_anticheat_testdata_feature.txt', sep='\t', encoding='utf-8')
    train = time_format(train) # 2019-06-03 2019-06-09
    test = time_format(test) # 2019-06-10

    train = md5_format(train)
    test = md5_format(test)
    #train['ip_info'] = train.apply(lambda x: ip_info(x['ip']), axis=1)
    #test['ip_info'] = test.apply(lambda x: ip_info(x['ip']), axis=1)
    print(train.columns)
    print(test.columns)
    train.to_csv(MODEL + 'train.csv', encoding='utf-8', index=False)
    test.to_csv(MODEL + 'test.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    model_input()


