import pandas as pd
import numpy as np
import time
import requests
import json
import sys
import datetime
import pickle
import warnings
warnings.filterwarnings("ignore")
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

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,tst_series=None,target=None,min_samples_leaf=1,smoothing=1,noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


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

def black_rate(train, test):
    print('rate ......')
    features = ['city','lan','os','osv','pro2', 'pkgname', 'adunitshowid', 'mediashowid', 'ver', 'make', 'model','nginxtime_hour','ip_cate',  'ntt',  'carrier', 'orientation','apptype']
    for fr in features:
        sta_df = train.groupby(fr).agg({'label': 'sum', 'sid': 'count'}).reset_index()
        sta_df[fr + '_rate'] = (sta_df['label']) / (sta_df['sid'])
        train = train.merge(sta_df[[fr, fr + '_rate']], how='left', on=fr)
        test = test.merge(sta_df[[fr, fr + '_rate']], how='left', on=fr)
        ctr = np.sum(train['label']) / train.shape[0]
        train[fr + '_rate'] = train[fr + '_rate'].fillna(ctr)
        test[fr + '_rate'] = test[fr + '_rate'].fillna(ctr)

    for fr in features:
        sta_df = train[train['nginxtime_date']!='2019-06-09'].groupby(fr).agg({'label': 'sum', 'sid': 'count'}).reset_index()
        sta_df[fr + '_rate_offline'] = (sta_df['label']) / (sta_df['sid'])
        train = train.merge(sta_df[[fr, fr + '_rate_offline']], how='left', on=fr)
        test = test.merge(sta_df[[fr, fr + '_rate_offline']], how='left', on=fr)
        ctr = np.sum(train['label']) / train.shape[0]
        train[fr + '_rate_offline'] = train[fr + '_rate'].fillna(ctr)
        test[fr + '_rate_offline'] = test[fr + '_rate'].fillna(ctr)

    return train, test

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
        'pkgname':[20,40,100],
        'adunitshowid':[2,5,13],
        'mediashowid':[4,12],
        'ver':[5,10,20,50,100],
        'lan':[10,20],
        'make':[1,5,10],
        'model':[1,2,5,10],
        'osv':[1,2],

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
    features = [ 'adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5']
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


def feature_encode(train, test):
    train_pro = train[train['nginxtime_date'] == '2019-06-03']
    for showdate in [ '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07', '2019-06-08', '2019-06-09']:
        print('encoding ' + showdate)
        df_test = train[train['nginxtime_date'] == showdate]
        df_train = train[train['nginxtime_date'] < showdate]
        for lf in  ['pkgname', 'adunitshowid', 'mediashowid', 'ver', 'city', 'lan', 'make', 'model', 'os', 'osv','pro2', 'ip_cate', 'ntt', 'carrier', 'orientation', 'province', 'apptype', ]:
            print(lf)
            trn, sub = target_encode(df_train[lf], df_test[lf], target=df_train.label, min_samples_leaf=100, smoothing=10,noise_level=0.01)
            df_test[lf + '_code'] = sub
        train_pro = train_pro.append(df_test)

    for lf in  ['pkgname', 'adunitshowid', 'mediashowid', 'ver', 'city', 'lan', 'make', 'model', 'os', 'osv','pro2', 'ip_cate', 'ntt', 'carrier', 'orientation', 'province', 'apptype', ]:
        print(lf)
        trn, sub = target_encode(train[lf], test[lf], target=train.label, min_samples_leaf=100, smoothing=10,noise_level=0.01)
        test[lf + '_code'] = sub
    return train_pro, test

def feature_concat(train, test):
    train_pro = train[train['nginxtime_date'] == '2019-06-03']
    predate = '2019-06-03'
    for showdate in [ '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07', '2019-06-08', '2019-06-09']:
        print('concating ...' + showdate)
        df_test = train[train['nginxtime_date'] == showdate]
        df_train = train[train['nginxtime_date'] == predate]
        for f1 in ['ip_net', 'reqrealip_net']:
            for f2 in ['pkgname', 'adunitshowid', 'mediashowid', 'apptype']:
                f1_f2 = df_train[df_train[f1] != 'empty'].groupby(f1).agg({f2: device_only}).reset_index()
                f1_f2.columns = [f1, f'{f1}_{f2}_only']
                #df_test = df_test.drop(columns=f'{f1}_{f2}_only')
                df_test = df_test.merge(f1_f2, on=f1, how='left')
                #df_test[f'{f1}_{f2}_only'] = df_test[f'{f1}_{f2}_only'].fillna(f1_f2[f'{f1}_{f2}_only'].mean())
                df_test[f'{f1}_{f2}_only'] = df_test[f'{f1}_{f2}_only'].fillna(0)
        predate = showdate
        train_pro = train_pro.append(df_test)

    for f1 in ['ip_net', 'reqrealip_net']:
        for f2 in ['pkgname', 'adunitshowid', 'mediashowid', 'apptype']:
            f1_f2 = df_test[df_test[f1] != 'empty'].groupby(f1).agg({f2: device_only}).reset_index()
            f1_f2.columns = [f1, f'{f1}_{f2}_only']
#            test = test.drop(columns=f'{f1}_{f2}_only')
            test = test.merge(f1_f2, on=f1, how='left')
            test[f'{f1}_{f2}_only'] = test[f'{f1}_{f2}_only'].fillna(0)
            #test[f'{f1}_{f2}_only'] = test[f'{f1}_{f2}_only'].fillna(f1_f2[f'{f1}_{f2}_only'].mean())
    return train_pro, test

def unique_count(index_col, feature, df_data, sub):
    if isinstance(index_col, list):
        name = "{0}_{1}_nq".format('_'.join(index_col), feature)
    else:
        name = "{0}_{1}_nq".format(index_col, feature)
    print(name+' unique......')
    gp1 = df_data.groupby(index_col)[feature].nunique().reset_index().rename(
        columns={feature: name})
    df_data = pd.merge(df_data, gp1, how='left', on=[index_col])
    sub = pd.merge(sub, gp1, how='left', on=[index_col])
    return df_data.fillna(0), sub.fillna(0)

def model_input():
    train = pd.read_csv(DATA + 'round1_iflyad_anticheat_traindata.txt', sep='\t', encoding='utf-8')
    test = pd.read_csv(DATA + 'round1_iflyad_anticheat_testdata_feature.txt', sep='\t', encoding='utf-8')
    # #处理orientation异常值
    # train.orientation[(train.orientation == 90) | (train.orientation == 2)] = 0
    # test.orientation[(test.orientation == 90) | (test.orientation == 2)] = 0
    # #carrier为-1就是未知0
    # train.carrier[train.carrier == -1] = 0
    # test.carrier[test.carrier == -1] = 0
    # #ntt 网络类型 0-未知, 1-有线网, 2-WIFI, 3-蜂窝网络未知, 4-2G, 5-3G, 6–4G
    # train.ntt[(train.ntt <= 0) | (train.ntt > 6)] = 0
    # train.ntt[(train.ntt <= 2) | (train.ntt >= 1)] = 1
    # test.ntt[(test.ntt <= 0) | (test.ntt > 6)] = 0
    # test.ntt[(test.ntt <= 2) | (test.ntt >= 1)] = 1

    fillna_features = ['ver', 'city', 'lan', 'make', 'model', 'osv']
    print('null fill......')
    for ff in fillna_features:
        train[ff] = train[ff].fillna('null')
        test[ff] = test[ff].fillna('null')
    print('feature cut......')
    train, test = feature_cut(train, test)
    train = time_format(train) # 2019-06-03 2019-06-09
    test = time_format(test) # 2019-06-10
    train = md5_format(train)
    test = md5_format(test)


    city_pro = pd.read_csv("./model/省份城市对应表.csv", encoding='gbk')
    city_pro.columns = ['pro2', 'city']
    train = train.merge(city_pro, how='left', on='city')
    test = test.merge(city_pro, how='left', on='city')
    train['pro2'] = train['pro2'].fillna('null')
    test['pro2'] = test['pro2'].fillna('null')
    train['ip_cate'] = train.apply(lambda x: ip_cate(x['ip']), axis=1)
    test['ip_cate'] = test.apply(lambda x: ip_cate(x['ip']), axis=1)
    train['reqrealip_cate'] = train.apply(lambda x: ip_cate(x['reqrealip']), axis=1)
    test['reqrealip_cate'] = test.apply(lambda x: ip_cate(x['reqrealip']), axis=1)
    train['ip_net'] = train.apply(lambda x: ip_network(x['ip'], x['ip_cate']), axis=1)
    test['ip_net'] = test.apply(lambda x: ip_network(x['ip'], x['ip_cate']), axis=1)
    train['reqrealip_net'] = train.apply(lambda x: ip_network(x['reqrealip'], x['reqrealip_cate']), axis=1)
    test['reqrealip_net'] = test.apply(lambda x: ip_network(x['reqrealip'], x['reqrealip_cate']), axis=1)


    train['hw'] = train['h']*train['w']
    test['hw'] = test['h']*test['w']
    train['dpi'] = train['h'].astype('str') + '_' + train['w'].astype('str')
    test['dpi'] = test['h'].astype('str') + '_' + test['w'].astype('str')
    #设备黑白名单处理
    train, test = device_blacklist(train, test)

    #黑名单率统计
    train, test = black_rate(train, test)

    #设备登录密度
    train, test = device_log(train, test)

    #设备 机型唯一性
    train, test = unique_count('adidmd5', 'model', train, test)
    train, test = unique_count('imeimd5', 'model', train, test)
    train, test = unique_count('macmd5', 'model', train, test)
    train, test = unique_count('openudidmd5', 'model', train, test)
    train, test = unique_count('idfamd5', 'model', train, test)
    # 设备 城市唯一性
    train, test = unique_count('adidmd5', 'city', train, test)
    train, test = unique_count('imeimd5', 'city', train, test)
    train, test = unique_count('macmd5', 'city', train, test)
    train, test = unique_count('openudidmd5', 'city', train, test)
    train, test = unique_count('idfamd5', 'city', train, test)
    # 设备 ip唯一性
    train, test = unique_count('adidmd5', 'ip', train, test)
    train, test = unique_count('imeimd5', 'ip', train, test)
    train, test = unique_count('macmd5', 'ip', train, test)
    train, test = unique_count('openudidmd5', 'ip', train, test)
    train, test = unique_count('idfamd5', 'ip', train, test)
    train, test = unique_count('adidmd5', 'reqrealip', train, test)
    train, test = unique_count('imeimd5', 'reqrealip', train, test)
    train, test = unique_count('macmd5', 'reqrealip', train, test)
    train, test = unique_count('openudidmd5', 'reqrealip', train, test)
    train, test = unique_count('idfamd5', 'reqrealip', train, test)

    train, test = unique_count('adidmd5', 'ip_net', train, test)
    train, test = unique_count('imeimd5', 'ip_net', train, test)
    train, test = unique_count('macmd5', 'ip_net', train, test)
    train, test = unique_count('openudidmd5', 'ip_net', train, test)
    train, test = unique_count('idfamd5', 'ip_net', train, test)

    train, test = unique_count('adidmd5', 'make', train, test)
    train, test = unique_count('imeimd5', 'make', train, test)
    train, test = unique_count('macmd5', 'make', train, test)
    train, test = unique_count('openudidmd5', 'make', train, test)
    train, test = unique_count('idfamd5', 'make', train, test)

    train, test = unique_count('adidmd5', 'pkgname', train, test)
    train, test = unique_count('imeimd5', 'pkgname', train, test)
    train, test = unique_count('macmd5', 'pkgname', train, test)
    train, test = unique_count('openudidmd5', 'pkgname', train, test)
    train, test = unique_count('idfamd5', 'pkgname', train, test)

    # #特征target_encode
    # train, test = feature_encode(train, test)
    #
    # #ip组合特征
    # train, test = feature_concat(train, test)

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
    # with open(MODEL + 'train.pk', 'rb') as train_f:
    #     train = pickle.load(train_f)
    # with open(MODEL + 'test.pk', 'rb') as test_f:
    #     test = pickle.load(test_f)
    # print(train[['adidmd5_model_nq']].head())
    #
    # train, test = feature_concat(train, test)
    # with open(MODEL + 'train.pk', 'wb') as train_f:
    #     pickle.dump(train, train_f)
    # with open(MODEL + 'test.pk', 'wb') as test_f:
    #     pickle.dump(test, test_f)