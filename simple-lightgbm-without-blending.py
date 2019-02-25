import time
import warnings
import logging
import gc
import lightgbm as lgb
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from utils import display_importances, rmse, set_logger
from datapipeline import get_df, FEATS_EXCLUDED
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df, test_df, num_folds, submission_file_name, stratified=False, debug=False):
    logger = logging.getLogger('lgbm_train')
    logger.info("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params = {
            'device': 'gpu',
            'task': 'train',
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'learning_rate': 0.01,
            'subsample': 0.718509060213284,
            'max_depth': 8,
            'top_rate': 0.8076614306859368,
            'num_leaves': 45,
            'min_child_weight': 59.174950161115106,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 17.018862389097798,
            'reg_lambda': 24.20636870149939,
            'colsample_bytree': 0.667864732544997,
            'min_split_gain': 8.021790442813048,
            'min_data_in_leaf': 30,
            'verbose': -1,
            'seed': int(2**n_fold),
            'bagging_seed': int(2**n_fold),
            'drop_seed': int(2**n_fold)
        }

        reg = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=['train', 'test'],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=100
        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logger.info('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # display importances
    display_importances(feature_importance_df)

    if not debug:
        # save submission file
        test_df.loc[:, 'target'] = sub_preds
        test_df = test_df.reset_index()
        test_df[['card_id', 'target']].to_csv(submission_file_name, index=False)

def main(submission_file_name, debug=False, logger=None):
    train_df, test_df = get_df(log_name='lgbm_train')
    logger.info("Run LightGBM with kfold")
    kfold_lightgbm(train_df, test_df, num_folds=11, stratified=False, debug=debug, submission_file_name=submission_file_name)


if __name__ == "__main__":
    timenow = time.strftime("%m%d-%H%M%S")
    logger = set_logger('lgbm_train', log_path='logs/logs_' + 'lgbm_train' + timenow)
    submission_file_name = "submission" + timenow+".csv"

    logger.info("Full model run")
    main(submission_file_name, debug=False, logger=logger)
