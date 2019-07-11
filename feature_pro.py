import pandas as pd
import numpy as np
import time
import requests
import json
import sys
import datetime
import pickle
DATA = './data/'
MODEL = './model/'

import re
ip_database = []
with open(DATA + 'CN_ips.txt') as ipfile:
    flines = ipfile.readlines()
    for l in flines:
        a = re.sub(' +', ' ', l.rstrip('\n'))
        ip_database.append(a.split(' ',3))
ip_database = pd.DataFrame(ip_database, columns=['ips','ipe','add','com'])
ip_database.to_csv(DATA+'cn_ips.csv', encoding='utf-8', index=False)

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

def ip_cate(ip):
    try:
        f_info = int(ip.split('.')[0])
    except:
        return 6
    if f_info >=1 and f_info<=126:
        return 1
    elif f_info == 127:
        return -1
    elif f_info >=128 and f_info<=191:
        return 2
    elif f_info >=192 and f_info<=223:
        return 3
    elif f_info >=224 and f_info<=239:
        return 4
    else:
        return 5

def ip_network(ip, ip_cate):
    if ip_cate==1:
        return ip.split('.')[0]
    elif ip_cate==2:
        return ip.split('.')[0] + '.' + ip.split('.')[1]
    elif ip_cate==3:
        return ip.split('.')[0] + '.' + ip.split('.')[1] + '.' + ip.split('.')[1]
    else:
        return -1

def device_blacklist_pro(df_train, df_test):
    wblist = ['adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5', 'ip_net', 'reqrealip_net', 'ip', 'reqrealip']
    for wbl in wblist:
        bl = df_train[(df_train[wbl] != 'empty') & (df_train['label'] == 1)].groupby(wbl).size().reset_index(name=wbl + 'bl')
        wl = df_train[(df_train[wbl] != 'empty') & (df_train['label'] == 0)].groupby(wbl).size().reset_index(name=wbl + 'wl')
        df_test = df_test.merge(bl, how='left', on=wbl)
        df_test = df_test.merge(wl, how='left', on=wbl)
        df_test[wbl + 'bl'] = df_test[wbl + 'bl'].fillna(0)
        df_test[wbl + 'wl'] = df_test[wbl + 'wl'].fillna(0)
        df_test[wbl + 'bl'] = df_test.apply(lambda x: 1 if x[wbl + 'bl'] > 0 else 0, axis=1)
        df_test[wbl + 'wl'] = df_test.apply(lambda x: 1 if x[wbl + 'wl'] > 0 else 0, axis=1)
    return df_test
def device_blacklist(train, test):
    train_pro = train[train['nginxtime_date'] == '2019-06-03']
    wblist = ['adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5', 'ip_net', 'reqrealip_net', 'ip', 'reqrealip']
    for wbl in wblist:
        train_pro[wbl + 'bl'] = train_pro['label']
        train_pro[wbl + 'wl'] = train_pro['label']
    #train_pro = pd.DataFrame()
    for showdate in ['2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07', '2019-06-08', '2019-06-09']:
        print(f'bwlist pro......{showdate}')
        df_test = train[train['nginxtime_date']==showdate]
        df_train = train[train['nginxtime_date']<showdate]
        df_test = device_blacklist_pro(df_train, df_test)
        train_pro = train_pro.append(df_test)

    test = device_blacklist_pro(train, test)
    return train_pro, test

def black_rate_pro(df_train, df_test):
    features = ['city','adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5',
                'pro2','adunitshowid','mediashowid','apptype','nginxtime',
                'ip_cate','reqrealip_cate',
                'dvctype','ntt','carrier','os','orientation', 'ip_net', 'reqrealip_net']
    for fr in features:
        sta_df = df_train.groupby(fr).agg({'label': 'sum', 'sid': 'count'}).reset_index()
        sta_df[fr+'_rate'] = (sta_df['label']+48) / (sta_df['sid']+100) #有效果
        df_test = df_test.merge(sta_df[[fr, fr+'_rate']], how='left', on=fr)
        ctr = np.sum(df_train['label']) / df_train.shape[0]
        df_test[fr+'_rate'] = df_test[fr+'_rate'].fillna(ctr)
    return df_test
def black_rate(train, test):
    train_pro = train[train['nginxtime_date'] == '2019-06-03']
    for showdate in ['2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07', '2019-06-08', '2019-06-09']:
        print(f'black rate  pro......{showdate}')
        df_test = train[train['nginxtime_date']==showdate]
        df_train = train[train['nginxtime_date']<showdate]
        df_test = black_rate_pro(df_train, df_test)
        train_pro = train_pro.append(df_test)
    test = black_rate_pro(train, test)
    return train_pro, test

def ip2decimalism(ip):
    dec_value = 0
    v_list = ip.split('.')
    v_list.reverse()
    t = 1
    try:
        for v in v_list:
            dec_value += int(v) * t
            t = t * (2 ** 8)
        return dec_value
    except:
        return -1

#对长尾的截取

def _cut(df_value, df_counts, count_list):
    for _c in count_list:
        if df_counts<=_c:
            return str(_c)+'_counts'
    return str(df_value)
def feature_cut(train, test):
    features_cutmap = {
        'pkgname':[1,2,3],
        'adunitshowid':[2,5,13],
        'mediashowid':[4,12],
        'ver':[1,2],
        'lan':[1,2],
        'make':[1],
        'model':[1,2],
        'osv':[1,2]
    }
    for ft in features_cutmap:
        df_counts = pd.DataFrame(train[ft].value_counts())
        df_counts.columns = [ft+'_counts']
        df_counts[ft] = df_counts.index
        train = train.merge(df_counts, on=ft, how='left')
        train[ft] = train.apply(lambda x: _cut(x[ft], x[ft+'_counts'], features_cutmap[ft]), axis=1)
        test = test.merge(df_counts, on=ft, how='left')
        test[ft+'_counts'] = test[ft+'_counts'].fillna(0)
        test[ft] = test.apply(lambda x: _cut(x[ft], x[ft + '_counts'], features_cutmap[ft]), axis=1)
        train.drop(columns=[ft+'_counts'],inplace=True)
        test.drop(columns=[ft + '_counts'], inplace=True)
    return train, test

def device_log_pro(df_train, df_test):
    features = [ 'adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5','ip', 'reqrealip']
    for fr in features:
        bl = df_train[(df_train[fr] != 'empty')].groupby(fr).size().reset_index(name=fr + '_counts')
        df_test = df_test.merge(bl, how='left', on=fr)
        df_test[fr+'_active'] = df_test.apply(lambda x: 1 if x[fr + '_counts'] > 0 else 0, axis=1)
    return df_test

def device_log(train, test):
    train_pro = train[train['nginxtime_date'] == '2019-06-03']
    dates = [ '2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07', '2019-06-08', '2019-06-09']
    for showdate_index in range(1,len(dates)):
        print(f'device_log  pro......{dates[showdate_index]}')
        df_test = train[train['nginxtime_date'] == dates[showdate_index]]
        df_train = train[train['nginxtime_date'] == dates[showdate_index-1]]
        df_test = device_log_pro(df_train, df_test)
        train_pro = train_pro.append(df_test)
    test = device_log_pro(train, test)
    return train_pro, test

def device_only(x):
    return len(list(set(list(x))))
def model_input():
    # train = pd.read_csv(DATA + 'round1_iflyad_anticheat_traindata.txt', sep='\t', encoding='utf-8')
    # test = pd.read_csv(DATA + 'round1_iflyad_anticheat_testdata_feature.txt', sep='\t', encoding='utf-8')
    # fillna_features = ['ver', 'city', 'lan', 'make', 'model', 'osv']
    # print('null fill......')
    # for ff in fillna_features:
    #     train[ff] = train[ff].fillna('null')
    #     test[ff] = test[ff].fillna('null')
    # print('feature cut......')
    # train, test = feature_cut(train, test)
    #
    # train = time_format(train) # 2019-06-03 2019-06-09
    # test = time_format(test) # 2019-06-10
    #
    # train = md5_format(train)
    # test = md5_format(test)
    #
    #
    # city_pro = pd.read_csv("./model/省份城市对应表.csv", encoding='gbk')
    # city_pro.columns = ['pro2', 'city']
    # train = train.merge(city_pro, how='left', on='city')
    # test = test.merge(city_pro, how='left', on='city')
    # train['pro2'] = train['pro2'].fillna('null')
    # test['pro2'] = test['pro2'].fillna('null')
    #
    # train['ipnum'] = train.apply(lambda x: ip2decimalism(x['ip']), axis=1)
    # test['ipnum'] = test.apply(lambda x: ip2decimalism(x['ip']), axis=1)
    # train['reqrealipnum'] = train.apply(lambda x: ip2decimalism(x['reqrealip']), axis=1)
    # test['reqrealipnum'] = test.apply(lambda x: ip2decimalism(x['reqrealip']), axis=1)
    #
    # train['ip_cate'] = train.apply(lambda x: ip_cate(x['ip']), axis=1)
    # test['ip_cate'] = test.apply(lambda x: ip_cate(x['ip']), axis=1)
    # train['reqrealip_cate'] = train.apply(lambda x: ip_cate(x['reqrealip']), axis=1)
    # test['reqrealip_cate'] = test.apply(lambda x: ip_cate(x['reqrealip']), axis=1)
    # train['ip_net'] = train.apply(lambda x: ip_network(x['ip'], x['ip_cate']), axis=1)
    # test['ip_net'] = test.apply(lambda x: ip_network(x['ip'], x['ip_cate']), axis=1)
    # train['reqrealip_net'] = train.apply(lambda x: ip_network(x['reqrealip'], x['reqrealip_cate']), axis=1)
    # test['reqrealip_net'] = test.apply(lambda x: ip_network(x['reqrealip'], x['reqrealip_cate']), axis=1)
    #
    #
    # train['hw'] = train['h']*train['w']
    # test['hw'] = test['h']*test['w']
    # #设备黑白名单处理
    # train, test = device_blacklist(train, test)
    #
    # #黑名单率统计
    # train, test = black_rate(train, test)
    #
    # #设备登录密度
    # train, test = device_log(train, test)

    #设备唯一性
    with open(MODEL + 'train.pk', 'rb') as train_f:
        train = pickle.load(train_f)
    with open(MODEL + 'test.pk', 'rb') as test_f:
        test = pickle.load(test_f)
    combine = pd.concat([train, test], axis=0)
    for f1,f2 in zip(['adidmd5', 'adidmd5', 'adidmd5', 'adidmd5', 'adidmd5'],['ip', 'idfamd5', 'openudidmd5', 'macmd5', 'imeimd5']):
        f1_f2 = combine[combine[f1]!='empty'].groupby(f1).agg({f2:device_only}).reset_index()
        f1_f2.columns=[f1,f1+f2+'_counts']
        train = train.merge(f1_f2, on=f1, how='left')
        test = test.merge(f1_f2, on=f1, how='left')
        train[f1+f2+'_counts'] =  train[f1+f2+'_counts'].fillna(f1_f2[f1+f2+'_counts'].mean())
        test[f1+f2+'_counts'] = test[f1+f2+'_counts'].fillna(f1_f2[f1+f2+'_counts'].mean())


    print(train.columns)
    print(test.columns)
    print(train.info())
    with open(MODEL + 'train.pk', 'wb') as train_f:
        pickle.dump(train, train_f)
    with open(MODEL + 'test.pk', 'wb') as test_f:
        pickle.dump(test, test_f)
    # train.to_csv(MODEL + 'train.csv', encoding='utf-8', index=False)
    # test.to_csv(MODEL + 'test.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    model_input()

'''
Index(['sid', 'pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype',
       'nginxtime', 'ip', 'city', 'province', 'reqrealip', 'adidmd5',
       'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5', 'dvctype', 'model',
       'make', 'ntt', 'carrier', 'os', 'osv', 'orientation', 'lan', 'h', 'w',
       'ppi', 'nginxtime_format', 'nginxtime_date', 'nginxtime_time',
       'nginxtime_hour', 'nginxtime_week', 'adidmd5_0', 'imeimd5_0',
       'idfamd5_0', 'openudidmd5_0', 'macmd5_0', 'pro2', 'ipnum',
       'reqrealipnum', 'ip_cate', 'reqrealip_cate', 'ip_net', 'hw',
       'adidmd5bl', 'adidmd5wl', 'imeimd5bl', 'imeimd5wl', 'idfamd5bl',
       'idfamd5wl', 'openudidmd5bl', 'openudidmd5wl', 'macmd5bl', 'macmd5wl',
       'city_rate', 'adidmd5_rate', 'imeimd5_rate', 'idfamd5_rate',
       'openudidmd5_rate', 'macmd5_rate', 'pro2_rate', 'adunitshowid_rate',
       'mediashowid_rate', 'apptype_rate', 'nginxtime_rate'],
only_features = ['adidmd5ip_counts', 'adidmd5idfamd5_counts', 'adidmd5openudidmd5_counts', 'adidmd5macmd5_counts', 'adidmd5imeimd5_counts']
'''
