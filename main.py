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
    #model_input()
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

    train['deviceid_sum'] = train['adidmd5_0'] + train['imeimd5_0'] + train['idfamd5_0'] + train['openudidmd5_0'] + train['macmd5_0']
    test['deviceid_sum'] = test['adidmd5_0'] + test['imeimd5_0'] + test['idfamd5_0'] + test['openudidmd5_0'] + test['macmd5_0']
    #特征处理
    offline = False
    if offline:
        model_train = train[(train['nginxtime_date'] != '2019-06-09') & (train['nginxtime_date'] > '2019-06-03')]
        model_test = train[train['nginxtime_date'] == '2019-06-09']
    else:
        model_train = train
        model_test = test

    label = 'label'
    rate_features = ['lan_rate_offline','os_rate_offline','osv_rate_offline', 'pkgname_rate_offline', 'adunitshowid_rate_offline', 'mediashowid_rate_offline',
                     'ver_rate_offline', 'make_rate_offline', 'model_rate_offline',  'ntt_rate_offline', 'orientation_rate_offline',
                     'apptype_rate_offline']
    rate_features = [i.replace('_offline','') for i in rate_features]
    wl_features = ['adidmd5bl', 'adidmd5wl',  'idfamd5bl', 'idfamd5wl',   'openudidmd5bl', 'openudidmd5wl',   'macmd5bl', 'macmd5wl',   'imeimd5bl', 'imeimd5wl',]
    active_features = [ 'adidmd5_active', 'idfamd5_active', 'imeimd5_active', 'macmd5_active', 'openudidmd5_active']
    label_features = ['pkgname','adunitshowid','mediashowid','ver','city','lan','make','model','os','osv','pro2', 'ip_cate', 'ntt', 'carrier', 'orientation', 'province','apptype',]
    only_features = []
    for de in ['adidmd5', 'imeimd5', 'macmd5', 'openudidmd5', 'idfamd5']:
        for el in ['model', 'city', 'ip', 'reqrealip', 'ip_net', 'make', 'pkgname']:
            only_features.append(f'{de}_{el}_nq')
    category_onehot_features = ['city','lan','os','osv','pro2', 'pkgname', 'adunitshowid', 'mediashowid', 'ver', 'make', 'model','nginxtime_hour','ip_cate',  'ntt',  'carrier', 'orientation','apptype','dpi']
    category_nullone_features = []
    numerical_features = ['h','w','ppi','hw','adidmd5_0','imeimd5_0','idfamd5_0','openudidmd5_0','macmd5_0', ] + wl_features + only_features + rate_features

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

        model_train[['sid', 'pre_proba']].to_csv(MODEL + f'train_track_{model_type}.csv', index=False, encoding='utf-8')
        model_test[['sid', 'pre_proba']].to_csv(MODEL + f'test_track_{model_type}.csv', index=False, encoding='utf-8')
        print(f'{train_f1} f1 score')

'''
train use times 2054.5476400852203  0.9426690926218865 f1 score  94.17  f1 train_score 0.9358662301138023 , f1 test_score 0.9342464228049663
f1 train_score 0.9480535374550796 , f1 test_score 0.9351981481224789  不带cv
f1 train_score 0.9358935120004898 , f1 test_score 0.9355122756172476   0.9408054251434533 f1 score   94.2218

f1 train_score 0.9485662798361463 , f1 test_score 0.9353841461724014 f1 train_score 0.9361220918749149 , f1 test_score 0.9355337516901847
f1 train_score 0.9370394692905323 , f1 test_score 0.9368188093602791  train use times 1988.2866513729095 0.9419362781666593 f1 score  
'''