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
    #特征处理
    offline = False
    if offline:
        model_train = train[(train['nginxtime_date'] != '2019-06-09') & (train['nginxtime_date'] > '2019-06-03')]
        model_test = train[train['nginxtime_date'] == '2019-06-09']
    else:
        model_train = train
        model_test = test

    label = 'label'
    rate_features = [ 'adidmd5_rate', 'imeimd5_rate', 'idfamd5_rate','openudidmd5_rate', 'macmd5_rate', 'ip_net_rate', 'ip_cate_rate']
    wl_features = ['adidmd5bl', 'adidmd5wl',  'idfamd5bl', 'idfamd5wl',   'openudidmd5bl', 'openudidmd5wl',   'macmd5bl', 'macmd5wl',   'imeimd5bl', 'imeimd5wl',]
    active_features = [ 'adidmd5_active', 'adidmd5_counts', 'idfamd5_active', 'idfamd5_counts', 'imeimd5_active', 'imeimd5_counts', 'macmd5_active', 'openudidmd5_active']
    only_features = ['adidmd5ip_counts', 'adidmd5idfamd5_counts', 'adidmd5openudidmd5_counts', 'adidmd5macmd5_counts','adidmd5imeimd5_counts']
    label_features = ['pkgname','adunitshowid','mediashowid','ver','city','lan','make','model','os','osv','pro2', 'ip_cate', 'ntt', 'carrier', 'orientation', 'province','apptype',]
    category_onehot_features = ['city','lan','os','osv','pro2', 'pkgname', 'adunitshowid', 'mediashowid', 'ver', 'make', 'model','nginxtime_hour','ip_cate',  'ntt',  'carrier', 'orientation','apptype',]
    category_nullone_features = []
    numerical_features = ['h','w','ppi','hw','adidmd5_0','imeimd5_0','idfamd5_0','openudidmd5_0','macmd5_0',


                          ] + wl_features
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

        model_train[['sid', 'pre_proba']].to_csv(MODEL + f'train_track_{model_type}.csv', index=False, encoding='utf-8')
        model_test[['sid', 'pre_proba']].to_csv(MODEL + f'test_track_{model_type}.csv', index=False, encoding='utf-8')
        print(f'{train_f1} f1 score')

'''
train use times 2054.5476400852203  0.9426690926218865 f1 score  94.17  f1 train_score 0.9358662301138023 , f1 test_score 0.9342464228049663

f1 train_score 0.9359606048381487 , f1 test_score 0.9345146057085545

train use times 1867.8405408859253
0.9426759425727612 f1 score

f1 train_score 0.9534761528795579 , f1 test_score 0.933866703770697
f1 train_score 0.9535149110065537 , f1 test_score 0.9338946077235631
f1 train_score 0.9473621274838466 , f1 test_score 0.9341076675722126
f1 train_score 0.9480535374550796 , f1 test_score 0.9351981481224789
f1 train_score 0.9358935120004898 , f1 test_score 0.9355122756172476   0.9408054251434533 f1 score   94.2218
'''