from feature_pro import *
from model import *
from sklearn.metrics import f1_score
import sys
import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    train = pd.read_csv(MODEL + 'train.csv', encoding='utf-8')
    test = pd.read_csv(MODEL + 'test.csv', encoding='utf-8')
    print(train.columns)
    fillna_features = ['city','lan','ver','model','make','osv','pro2']
    for ff in fillna_features:
        train[ff] = train[ff].fillna('null')
        test[ff] = test[ff].fillna('null')

    train['dvctype'] = train['dvctype'].astype(int)
    test['dvctype'] = test['dvctype'].astype(int)
    train['ntt'] = train['ntt'].astype(int)
    test['ntt'] = test['ntt'].astype(int)
    train['carrier'] = train['carrier'].astype(int)
    test['carrier'] = test['carrier'].astype(int)

    train['adid_imei'] = train['adidmd5_0'] + train['imeimd5_0']
    train['adid_idfa'] = train['adidmd5_0'] + train['idfamd5_0']
    train['adid_openudid'] = train['adidmd5_0'] + train['openudidmd5_0']
    train['adid_mac'] = train['adidmd5_0'] + train['macmd5_0']

    offline = False
    if offline:
        model_train = train[(train['nginxtime_date'] != '2019-06-09') & (train['nginxtime_date'] > '2019-06-03')]
        model_test = train[train['nginxtime_date'] == '2019-06-09']
    else:
        model_train = train
        model_test = test
    label = 'label'
    """
    基本信息 sid label
    媒体信息 pkgname ver adunitshowid mediashowid apptype 
    时间  nginxtime_hour
    ip信息  city province
    设备信息 
    """
    features = {
        'category_features':['pkgname','adunitshowid','mediashowid','ver',
            'city',
            'lan','make','model','os','osv',
            'pro2'],
        'numerical_features':['apptype',
                              'province','ip_cate',
                              'nginxtime_hour','nginxtime_week',
                              'dvctype','ntt','carrier','orientation','h','w','ppi','hw',
                              'adidmd5_0','imeimd5_0','idfamd5_0','openudidmd5_0','macmd5_0',
                              'city_rate', 'adidmd5_rate', 'imeimd5_rate', 'idfamd5_rate',
                              'openudidmd5_rate', 'macmd5_rate', 'pro2_rate', 'adunitshowid_rate',
                              'mediashowid_rate', 'apptype_rate','nginxtime_rate'

                              ],
        'onehot_features': ['pkgname','adunitshowid','mediashowid','ver',
            'city',
            'lan','make','model','os','osv',
            'pro2'],
        'label':'label'
    }
    model_type='lgb'
    t1 = time.time()
    pre_proba = class_model(model_train, model_test, features, model_type,class_num=2,cv=False)

    print(f"train use times {time.time() - t1}")
    model_test['pre_proba'] = pre_proba[:, 1]
    if offline:
        #model_test['pre_proba'] = model_test.apply(lambda x: add_proba(x['pre_proba'], x['bl_sum']), axis=1)
        model_test['pre_label'] = model_test.apply(lambda x: 1 if x['pre_proba'] > 0.5 else 0, axis=1)
        f1 = f1_score(model_test['label'], model_test['pre_label'])
        print('f1 score',f1)
    else:
        model_test['label'] = model_test.apply(lambda x: 1 if x['pre_proba']>0.5 else 0, axis=1)
        model_test[['sid','label']].to_csv('submit/submission.csv', index=False, encoding='utf-8')

'''
0.7547364405702094 time:340.9751136302948
0.9053778269984885 time:570.2333481311798 

f1 score 0.9336021114320747  93.99
f1 score 0.9337018492927132  94.10357

f1 score 0.9339196643185639
f1 score 0.9339468132356207
'''

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
'''