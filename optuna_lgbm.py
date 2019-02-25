# https://github.com/MitsuruFujiwara/National-Parks/blob/master/src/optuna_lgbm.py
import time
import gc
import lightgbm
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from datapipeline import get_df, FEATS_EXCLUDED
from utils import set_logger
################################################################################
# optunaによるhyper parameter最適化
# 参考: https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
################################################################################

logger = set_logger('optuna_lgbm', log_path='logs/logs_' + 'optuna_lgbm' + time.strftime("%m%d-%H%M%S"))
NUM_FOLDS=5
# split test & train
TRAIN_DF, _ = get_df(log_name='optuna_lgbm')
FEATS = [f for f in TRAIN_DF.columns if f not in FEATS_EXCLUDED]

def objective(trial):
    lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                                  label=TRAIN_DF['target'],
                                  free_raw_data=False
                                  )

    params = {'objective': 'regression',
              'metric': 'rmse',
              'verbosity': -1,
              "learning_rate": 0.01,
              'device': 'gpu',
              'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
              'subsample': trial.suggest_uniform('subsample', 0.5, 1),
              'max_depth': trial.suggest_int('max_depth', 5, 25),
              'top_rate': trial.suggest_uniform('top_rate', 0.5, 1),
              'num_leaves': trial.suggest_int('num_leaves', 16, 96),
              'min_child_weight': trial.suggest_uniform('min_child_weight', 25, 75),

            #   lambda_l1 ： default = 0.0, type = double, aliases: reg_alpha, constraints: lambda_l1 >= 0.0
            #   L1 regularization
              'reg_alpha': trial.suggest_uniform('reg_alpha', 5, 25),
            #   lambda_l2 ： default = 0.0, type = double, aliases: reg_lambda, lambda, constraints: lambda_l2 >= 0.0
            #   L2 regularization
              'reg_lambda': trial.suggest_uniform('reg_lambda', 5, 25),

              'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.25, 1),
              'min_split_gain': trial.suggest_uniform('min_split_gain', 5, 15),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 16, 48),
              'seed': 42,
              'bagging_seed': 42,
              'drop_seed': 42
              }
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if params['boosting_type'] == 'goss':
        # Input: I: training data, d: iterations
        # Input: a: sampling ratio of large gradient data
        # Input: b: sampling ratio of small gradient data
        # topN ← a × len(I) , randN ← b × len(I)
        params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])

    folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=47)
    clf = lightgbm.cv(params=params,
                      train_set=lgbm_train,
                      metrics=['rmse'],
                      nfold=NUM_FOLDS,
                      folds=folds.split(TRAIN_DF[FEATS], np.zeros(len(TRAIN_DF['target']))),
                      num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                      early_stopping_rounds=200,
                      verbose_eval=100,
                      seed=47
                     )
    gc.collect()
    return clf['rmse-mean'][-1]

if __name__ == '__main__':

    logger.info('optuna LightGBM start.')
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    # save result
    hist_df = study.trials_dataframe()
    hist_df.to_csv("logs/optuna_result_lgbm.csv")
    vals = ''
    vals += 'Number of finished trials: {}\n'.format(len(study.trials))
    vals += 'Best trial:\n'

    trial = study.best_trial

    vals += '  Value: {}\n'.format(trial.value)
    vals += '  Params:\n'

    for key, value in trial.params.items():
        vals += ('    {}: {}\n'.format(key, value))
    logger.info(vals)


    logger.info('optuna LightGBM finished.')
