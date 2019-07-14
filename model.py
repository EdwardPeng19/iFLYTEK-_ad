import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from scipy import sparse
from scipy.sparse import csr_matrix
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'n_estimators': 10000,
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 63,
    'max_depth': 7,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'random_state': 4590,
    'n_jobs': -1
}
xgb_params = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'n_estimators': 10000,
        'min_child_weight': 3,
        'gamma': 0,
        'silent': True,
        'n_jobs': -1,
        'random_state': 4590,
        'reg_alpha': 2,
        'reg_lambda': 0.1,
        'alpha': 1,
        'verbose': 1
    }

def multi_column_LabelEncoder(df,columns,rename=True):
    le = LabelEncoder()
    for column in columns:
        print(column,"LabelEncoder......")
        le.fit(df[column])
        df[column+"_index"] = le.transform(df[column])
        if rename:
            df.drop([column], axis=1, inplace=True)
            df.rename(columns={column+"_index":column}, inplace=True)
    print('LabelEncoder Successfully!')
    return df

def reg_model(model_train, test, train_label, model_type, onehot_features, label_features, features):
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    model_train.reset_index(inplace=True)
    train_label.index = range(len(train_label))
    test.reset_index(inplace=True)
    if model_type == 'rf':
        model_train.fillna(0, inplace=True)

    combine = pd.concat([model_train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, label_features, rename=True)
    #one hot 处理
    if onehot_features != []:
        onehoter = OneHotEncoder()
        X_onehot = onehoter.fit_transform(combine[onehot_features])
        train_x_onehot = X_onehot.tocsr()[:model_train.shape[0]].tocsr()
        test_x_onehot = X_onehot.tocsr()[model_train.shape[0]:].tocsr()

        train_x_original = combine[features][:model_train.shape[0]]
        test_x_original = combine[features][model_train.shape[0]:]

        train_x = sparse.hstack((train_x_onehot, train_x_original)).tocsr()
        test_x = sparse.hstack((test_x_onehot, test_x_original)).tocsr()
    else:
        tmp = combine[features].astype(np.float32)
        train_x = csr_matrix(tmp[:model_train.shape[0]].values)
        test_x = csr_matrix(tmp[model_train.shape[0]:].values)


    train_y = train_label

    n_fold = 5
    count_fold = 0
    preds_list = list()
    oof = np.zeros(train_x.shape[0])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y)
    for train_index, vali_index in kfold:
        print("training......fold",count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y.loc[vali_index]
        if model_type == 'lgb':
            dtrain = lgb.Dataset(k_x_train, k_y_train)
            dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False, eval_metric="l2")
            k_pred = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
            pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)
        elif model_type == 'xgb':
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model = xgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False)
            k_pred = xgb_model.predict(k_x_vali)
            pred = xgb_model.predict(test_x)
        elif model_type == 'rf':
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, criterion="mae",n_jobs=-1,random_state=2019)
            model = rf_model.fit(k_x_train, k_y_train)
            k_pred = rf_model.predict(k_x_vali)
            pred = rf_model.predict(test_x)
        preds_list.append(pred)
        oof[vali_index] = k_pred
    print(pd.DataFrame({
        'column': features,
        'importance': lgb_model.feature_importance(),
    }).sort_values(by='importance'))
    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds = list(preds_df.mean(axis=1))


    return preds


def class_model(train, test, features_map, model_type='lgb', class_num=2, cv=True):
    label = features_map['label']
    label_features = features_map['label_features']
    category_onehot_features = features_map['category_features1']
    category_features = features_map['category_features2']
    numerical_features = features_map['numerical_features']
    combine = pd.concat([train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, label_features, rename=True)
    combine.reset_index(inplace=True)
    onehoter = OneHotEncoder()
    X_onehot = onehoter.fit_transform(combine[category_onehot_features])
    train_x_onehot = X_onehot.tocsr()[:train.shape[0]].tocsr()
    test_x_onehot = X_onehot.tocsr()[train.shape[0]:].tocsr()
    train_x_original = combine[numerical_features+category_features][:train.shape[0]]
    test_x_original = combine[numerical_features+category_features][train.shape[0]:]
    train_x = sparse.hstack((train_x_onehot, train_x_original)).tocsr()
    test_x = sparse.hstack((test_x_onehot, test_x_original)).tocsr()
    train_y = combine[label][:train.shape[0]]

    # train_x = combine.loc[:train.shape[0]-1]
    # test_x = combine.loc[train.shape[0]:]
    # train_x[label] = train_x[label].astype(np.int)
    # features = category_features + numerical_features
    #
    # train_x = train_x[features].values
    # test_x = test_x[features].values



    #模型训练
    clf = lgb.LGBMClassifier(
        objective='binary',
        learning_rate=0.05,
        n_estimators=5000,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=2019,
        reg_alpha=1,
        reg_lambda=5,
        n_jobs=6
    )
    cxgb = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.01,
        n_estimators=100,
        subsample=0.8,
        random_state=2019,
        n_jobs=-1
    )
    if cv:
        n_fold = 5
        print(train.shape[0])
        result = np.zeros((test.shape[0],class_num))
        oof = np.zeros((train.shape[0],class_num))
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2019)
        kfold = skf.split(train_x, train_y)
        count_fold = 0
        for train_index, vali_index in kfold:
            print("training......fold",count_fold)
            count_fold = count_fold + 1
            k_x_train = train_x[train_index]
            k_y_train = train_y.loc[train_index]
            k_x_vali = train_x[vali_index]
            k_y_vali = train_y.loc[vali_index]
            if model_type == 'lgb':
                clf.fit(k_x_train, k_y_train,eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],early_stopping_rounds=200, verbose=False)
                test_pred_proba = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)
                val_pred_proba = clf.predict_proba(k_x_vali, num_iteration=clf.best_iteration_)
            elif model_type == 'xgb':
                cxgb.fit(k_x_train, k_y_train,eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],early_stopping_rounds=200, verbose=False)
                test_pred_proba = cxgb.predict_proba(test_x)
                val_pred_proba = cxgb.predict_proba(k_x_vali)
            result = result + test_pred_proba
            oof[vali_index] = val_pred_proba
        result = result/n_fold
    else:
        print(train_x.shape, train_y.shape)
        clf.fit(train_x, train_y)
        #test_pred = clf.predict(test_x)
        test_pred_proba = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)
        train_pred_proba = clf.predict_proba(train_x, num_iteration=clf.best_iteration_)
        result = test_pred_proba
        oof = train_pred_proba

    return oof,result

