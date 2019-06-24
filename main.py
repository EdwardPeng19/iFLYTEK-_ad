from feature_pro import *
from model import *
from sklearn.metrics import f1_score

if __name__ == "__main__":
    train = pd.read_csv(MODEL + 'train.csv', encoding='utf-8')
    test = pd.read_csv(MODEL + 'test.csv', encoding='utf-8')
    train['city'] = train['city'].fillna('null')
    test['city'] = test['city'].fillna('null')

    train['lan'] = train['lan'].fillna('null')
    test['lan'] = test['lan'].fillna('null')
    label = 'label'
    """
    基本信息 sid label
    媒体信息 pkgname ver adunitshowid mediashowid apptype 
    时间  nginxtime_hour
    ip信息  city province
    设备信息 
    """
    features = {
        'category_features':['city','pkgname', 'adunitshowid', 'mediashowid','lan'],
        'numerical_features':['apptype','province','nginxtime_hour',
                              'dvctype','ntt','carrier','orientation','h','w','ppi'],
        'label':'label'
    }
    offline = False
    if offline:
        model_train = train[train['nginxtime_date']!='2019-06-09']
        model_test = train[train['nginxtime_date']=='2019-06-09']
    else:
        model_train = train
        model_test = test

    model_type='lgb'
    t1 = time.time()
    pre_label = class_model(model_train, model_test, features, model_type,False)


    if offline:
        print(f"train use times {time.time() - t1}")
        print((model_test['label'], pre_label))
        f1 = f1_score(model_test['label'], pre_label)
        print(f1)
    else:
        model_test['label'] = pre_label
        model_test[['sid','label']].to_csv('submit/submission.csv', index=False, encoding='utf-8')

'''
0.7547364405702094 time:340.9751136302948
0.9053778269984885 time:570.2333481311798
'''