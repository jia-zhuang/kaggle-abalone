from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import gc
from datetime import datetime
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pandas as pd
import fire

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def lgb_cv(learning_rate, max_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, reg_lambda, data, target, feature_name, categorical_feature):
    folds = KFold(n_splits=4, shuffle=True, random_state=11)
    oof = np.zeros(data.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):
        # print(f'fold: {fold_}')
        trn_data = lgb.Dataset(data[trn_idx], label=target[trn_idx], feature_name=feature_name, categorical_feature=categorical_feature)
        val_data = lgb.Dataset(data[val_idx], label=target[val_idx], feature_name=feature_name, categorical_feature=categorical_feature)
        param = {
            # general parameters
            'num_threads': 8,
            'verbose': -1,
            'objective': 'regression',
            'metric': 'rmse',
            'bagging_freq': 1,
            'early_stopping_rounds': 600,
            # tuning parameters
            'learning_rate': learning_rate,
            'max_leaves': round(max_leaves),
            'min_data_in_leaf': round(min_data_in_leaf),
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'reg_lambda': reg_lambda
        }
        clf = lgb.train(param, trn_data, 10000, valid_sets=[trn_data, val_data])
        oof[val_idx] = clf.predict(data[val_idx], num_iteration=clf.best_iteration)
        oof = oof.clip(0, 30)
        del clf, trn_idx, val_idx
        gc.collect()
    return -mean_squared_log_error(target, oof, squared=False)


def xgb_cv(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, data, target):
    folds = KFold(n_splits=4, shuffle=True, random_state=11)
    oof = np.zeros(data.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):
        # print(f'fold: {fold_}')
        trn_data = xgb.DMatrix(data[trn_idx], label=target[trn_idx])
        val_data = xgb.DMatrix(data[val_idx], label=target[val_idx])
        param = {
            # general parameters
            'nthread': 16,
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            # tuning parameters
            'learning_rate': learning_rate,
            'max_depth': round(max_depth),
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
        }
        clf = xgb.train(param, trn_data, 10000, evals=[(trn_data, 'train'), (val_data, 'valid')], verbose_eval=False, early_stopping_rounds=600)
        oof[val_idx] = clf.predict(val_data, iteration_range=(0, clf.best_iteration + 1))
        oof = oof.clip(0, 30)
        del clf, trn_idx, val_idx
        gc.collect()
    return -mean_squared_log_error(target, oof, squared=False)


def cb_cv(learning_rate, max_depth, min_data_in_leaf, subsample, colsample_bylevel, reg_lambda, data, target, feature_name, categorical_feature):
    folds = KFold(n_splits=4, shuffle=True, random_state=11)
    oof = np.zeros(data.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):
        # print(f'fold: {fold_}')
        trn_data = cb.Pool(data.iloc[trn_idx], label=target[trn_idx], feature_names=feature_name, cat_features=categorical_feature)
        val_data = cb.Pool(data.iloc[val_idx], label=target[val_idx], feature_names=feature_name, cat_features=categorical_feature)
        param = {
            # general parameters
            'thread_count': 8,
            # 'verbosity': 2,
            'objective': 'RMSE',
            'eval_metric': 'RMSE',
            # tuning parameters
            'learning_rate': learning_rate,
            'max_depth': round(max_depth),
            'min_data_in_leaf': round(min_data_in_leaf),
            'subsample': subsample,
            'colsample_bylevel': colsample_bylevel,
            'reg_lambda': reg_lambda,
        }
        clf = cb.train(trn_data, param, iterations=10000, evals=val_data, verbose_eval=False, early_stopping_rounds=600)
        oof[val_idx] = clf.predict(val_data, ntree_end=clf.best_iteration_)
        oof = oof.clip(0, 30)
        del clf, trn_idx, val_idx
        gc.collect()
    return -mean_squared_log_error(target, oof, squared=False)


def optimize_lgb(data, target, feature_name='auto', categorical_feature='auto'):
    def lgb_crossval(learning_rate, max_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, reg_lambda):
        return lgb_cv(learning_rate, max_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, reg_lambda, data, target, feature_name, categorical_feature)
    
    optimizer = BayesianOptimization(lgb_crossval, {
        'learning_rate': (0.005, 1),
        'max_leaves': (2, 200),
        'min_data_in_leaf': (2, 200),
        'bagging_fraction': (0.2, 1.0),
        'feature_fraction': (0.2, 1.0),
        'reg_lambda': (0, 10)
    })

    start_time = timer()
    optimizer.maximize(init_points=5, n_iter=200)
    timer(start_time)
    print("Final result:", optimizer.max)


def optimize_xgb(data, target):
    def xgb_crossval(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda):
        return xgb_cv(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, data, target)
    
    optimizer = BayesianOptimization(xgb_crossval, {
        'learning_rate': (0.005, 1),
        'max_depth': (2, 10),
        'min_child_weight': (2, 200), 
        'subsample': (0.2, 1.0),
        'colsample_bytree': (0.2, 1.0),
        'reg_lambda': (0, 10)
    })

    start_time = timer()
    optimizer.maximize(init_points=5, n_iter=200)
    timer(start_time)
    print("Final result:", optimizer.max)


def optimize_cb(data, target, feature_name='auto', categorical_feature='auto'):
    def cb_crossval(learning_rate, max_depth, min_data_in_leaf, subsample, colsample_bylevel, reg_lambda):
        return cb_cv(learning_rate, max_depth, min_data_in_leaf, subsample, colsample_bylevel, reg_lambda, data, target, feature_name, categorical_feature)
    
    optimizer = BayesianOptimization(cb_crossval, {
        'learning_rate': (0.005, 1),
        'max_depth': (2, 10),
        'min_data_in_leaf': (2, 200),
        'subsample': (0.2, 1.0),
        'colsample_bylevel': (0.2, 1.0),
        'reg_lambda': (0, 10)
    })

    start_time = timer()
    optimizer.maximize(init_points=5, n_iter=200)
    timer(start_time)
    print("Final result:", optimizer.max)


def prepare_train_data():
    train_df = pd.read_csv('../input/train.csv')
    Sex2code = {'I': 0, 'M': 1, 'F': 2}
    train_df['Sex_code'] = train_df['Sex'].map(Sex2code)
    train_cols = [c for c in train_df.columns if c not in {'Rings', 'Sex'}]
    target_col = 'Rings'
    return train_df, train_cols, target_col
    

def main(model_type):
    ''' model_type: xgboost, lightgbm or catboost
    '''
    train_df, train_cols, target_col = prepare_train_data()

    if model_type == 'xgb':
        optimize_xgb(train_df[train_cols].values, train_df[target_col].values)
    elif model_type == 'lgb':
        optimize_lgb(train_df[train_cols].values, train_df[target_col].values, feature_name=train_cols, categorical_feature=['Sex_code'])
    elif model_type == 'cb':
        optimize_cb(train_df[train_cols], train_df[target_col].values, feature_name=train_cols, categorical_feature=['Sex_code'])
    else:
        raise ValueError(f'Invalid model_type: {model_type}')

if __name__ == "__main__":
    fire.Fire(main)
