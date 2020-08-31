from pycaret.regression import *


def regr(data, target, metric='R2', session_id=42, model_name='model', turbo=True, top_n=5, use_gpu=False):
    'Apply pycaret for fast regression model building'

    '''
    setup default: silent = True, html = False, log_experiment=True, log_plots = True, log_profile = True, log_data = True
    must specify: data, target, session_id
    specify feature types: categorical_features, numeric_features, ordinal_features, date_features, ignore_features
    specify group feature: group_features, group_names
    feature engineering option 1: pca, pca_method, pca_components, combine_rare_levels, rare_level_threshold, bin_numeric_features, remove_outliers, outliers_threshold
    feature engineering option 2: polynomial_features, polynomial_degree, trigonometry_features, polynomial_threshold
    feature engineering option 3: normalize, normalize_method, transformation, transformation_method, create_clusters
    target engineering option: transform_target, transform_target_method
    hardware: use_gpu
    '''
    print('Setup')
    regr_exp = setup(data=data, target=target, session_id=session_id,
                     silent=True, html=False, verbose=False, use_gpu=use_gpu,
                     log_experiment=True, log_plots=True, log_profile=True, log_data=True)

    '''
    sort: 'MAE', 'MSE', 'RMSE', 'R2', 'RMSL', 'MAPE'
    turbo: When turbo is set to True, it excludes estimators that have longer training times.
    '''
    print('Top models')
    Top_models = compare_models(exclude=None, include=None, fold=10, round=4, sort=metric,
                                n_select=top_n, budget_time=0, turbo=turbo, verbose=False)

    '''
    fine tune top 5 model
    n_iter: Number of iterations within the Random Grid Search.
    '''
    print('Fine tuning')
    Tuned_models = [tune_model(model, verbose=False) for model in Top_models]

    '''
    blending and staking
    optimize: : 'MAE', 'MSE', 'RMSE', 'R2', 'RMSL', 'MAPE'
    meta_model: logistic, xgboost... etc
    '''
    print('Blending')
    blend_model = blend_models(estimator_list='All', fold=10, round=4,
                               choose_better=True, optimize=metric, turbo=turbo, verbose=False)
    print('Stacking')
    stacked_model = stack_models(estimator_list=Top_models, meta_model=None, fold=10, round=4,
                                 restack=True, choose_better=True, optimize=metric, verbose=False)

    '''
    save the finalized best model based on holdout set metric
    '''
    print('Finalizing')
    best = automl(optimize=metric, use_holdout=True)
    final = finalize_model(best)
    save_model(final, model_name, model_only=False, verbose=False)

    return final, get_logs(save=True)
