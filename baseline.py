from data_pre import data_pre
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
train, test = data_pre()


label = train['收率']
test_id = test['样本id']
del test['样本id']
del test['收率']
del train['样本id']
del train['收率']


train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

# 五折交叉验证
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))


param = {'num_leaves': 120,
         'min_data_in_leaf': 30,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}

X_train = train.values
y_train = label.values
X_test = test.values
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=200,
                    early_stopping_rounds=100)
    oof[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, y_train)))

# 提交结果
sub_df = pd.read_csv('input/jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = predictions
sub_df.to_csv("sub.csv", index=False, header=None)