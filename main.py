from feature_pro import *
from model import *
from sklearn.metrics import f1_score
import sys
import warnings
warnings.filterwarnings('ignore')

def cut_hour(x):
    if x>=12 and x<=21:
        return 1
    elif x>=22 and x<=11 :
        return 2
    else:
        return 3
if __name__ == "__main__":
    # train = pd.read_csv(MODEL + 'train.csv', encoding='utf-8')
    # test = pd.read_csv(MODEL + 'test.csv', encoding='utf-8')
    with open(MODEL + 'train.pk', 'rb') as train_f:
        train = pickle.load(train_f)
    with open(MODEL + 'test.pk', 'rb') as test_f:
        test = pickle.load(test_f)
    train['nginxtime_hour'] = train['nginxtime_hour'].astype(float)
    test['nginxtime_hour'] = test['nginxtime_hour'].astype(float)
    train['nginxtime_hour'] = train.apply(lambda  x: cut_hour(x['nginxtime_hour']), axis=1)
    test['nginxtime_hour'] = test.apply(lambda x: cut_hour(x['nginxtime_hour']), axis=1)
    #特征处理
    offline = False
    if offline:
        model_train = train[(train['nginxtime_date'] != '2019-06-09') & (train['nginxtime_date'] > '2019-06-03')]
        model_test = train[train['nginxtime_date'] == '2019-06-09']
    else:
        model_train = train
        model_test = test

    label = 'label'
    rate_features = [ 'adidmd5_rate', 'imeimd5_rate', 'idfamd5_rate',
                          'openudidmd5_rate', 'macmd5_rate',]
    wl_features = ['adidmd5bl', 'adidmd5wl',  'idfamd5bl', 'idfamd5wl',   'openudidmd5bl', 'openudidmd5wl',   'macmd5bl', 'macmd5wl',   'imeimd5bl', 'imeimd5wl',]
    label_features = ['pkgname','adunitshowid','mediashowid','ver','city','lan','make','model','os','osv','pro2']

    category_onehot_features = ['city','lan','os','osv','pro2', 'pkgname', 'adunitshowid', 'mediashowid', 'ver', 'make', 'model',
                                'nginxtime_hour',
                                ]
    category_nullone_features = ['apptype', 'dvctype',   'ip_cate',  'ntt',  'carrier', 'orientation', 'province']
    numerical_features = ['h','w','ppi','hw','adidmd5_0','imeimd5_0','idfamd5_0','openudidmd5_0','macmd5_0',


                          ] + rate_features + wl_features
    #
    features = {
        'label_features':label_features,
        'category_features1':category_onehot_features,
        'category_features2':category_nullone_features,
        'numerical_features':numerical_features,
        'label':'label'
    }
    model_type='lgb'
    t1 = time.time()
    train_pre_proba,test_pre_proba = class_model(model_train, model_test, features, model_type,class_num=2,cv=True)

    print(f"train use times {time.time() - t1}")
    model_test['pre_proba'] = test_pre_proba[:, 1]
    model_train['pre_proba'] = train_pre_proba[:,1]
    model_train['pre_label'] = model_train.apply(lambda x: 1 if x['pre_proba'] > 0.5 else 0, axis=1)
    train_f1 = f1_score(model_train['label'], model_train['pre_label'])
    if offline:
        model_test['pre_label'] = model_test.apply(lambda x: 1 if x['pre_proba'] > 0.5 else 0, axis=1)
        test_f1 = f1_score(model_test['label'], model_test['pre_label'])
        print(f'f1 train_score {train_f1} , f1 test_score {test_f1}')
    else:
        model_test['label'] = model_test.apply(lambda x: 1 if x['pre_proba']>0.5 else 0, axis=1)
        model_test[['sid','label']].to_csv('submit/submission.csv', index=False, encoding='utf-8')

        model_train[['sid', 'pre_proba']].to_csv(MODEL + 'train_track.csv', index=False, encoding='utf-8')
        model_test[['sid', 'pre_proba']].to_csv(MODEL + 'test_track.csv', index=False, encoding='utf-8')
        print(f'{train_f1} f1 score')

'''
Index(['adidmd5', 'adidmd5_0', 'adidmd5_rate', 'adidmd5bl', 'adidmd5wl',
       'adunitshowid', 'adunitshowid_rate', 'apptype', 'apptype_rate',
       'carrier', 'carrier_rate', 'city', 'city_rate', 'dvctype',
       'dvctype_rate', 'h', 'hw', 'idfamd5', 'idfamd5_0', 'idfamd5_rate',
       'idfamd5bl', 'idfamd5wl', 'imeimd5', 'imeimd5_0', 'imeimd5_rate',
       'imeimd5bl', 'imeimd5wl', 'ip', 'ip_cate', 'ip_cate_rate', 'ip_net',
       'ipnum', 'label', 'lan', 'macmd5', 'macmd5_0', 'macmd5_rate',
       'macmd5bl', 'macmd5wl', 'make', 'mediashowid', 'mediashowid_rate',
       'model', 'nginxtime', 'nginxtime_date', 'nginxtime_format',
       'nginxtime_hour', 'nginxtime_rate', 'nginxtime_time', 'nginxtime_week',
       'ntt', 'ntt_rate', 'openudidmd5', 'openudidmd5_0', 'openudidmd5_rate',
       'openudidmd5bl', 'openudidmd5wl', 'orientation', 'orientation_rate',
       'os', 'os_rate', 'osv', 'pkgname', 'ppi', 'pro2', 'pro2_rate',
       'province', 'reqrealip', 'reqrealip_cate', 'reqrealip_cate_rate',
       'reqrealipnum', 'sid', 'ver', 'w'],
       
       carrier_rate  dvctype_rate  ip_cate_rate   ntt_rate    orientation_rate   os_rate   pro2_rate  reqrealip_cate_rate
'''

'''
f1 train_score 0.9356451000385325 , f1 test_score 0.9347783771759078  94.16   0.9426083947858783 f1 score

'''