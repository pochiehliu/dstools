from pycaret.regression import *


def regr(data, target, metric='R2', seed=42, model_name='model',
         turbo=True, top_n=5, gpu=False, grid=10, ensemble=False, test_data=None):
    'Apply pycaret for fast regression model building'

    print('Setup')
    regr_exp = setup(data=data, target=target, session_id=seed, use_gpu=gpu,
                     silent=True, html=False, verbose=False,
                     log_experiment=True, log_plots=True, log_profile=True, log_data=True)

    '''
    metric option: 'MAE', 'MSE', 'RMSE', 'R2', 'RMSL', 'MAPE'
    '''
    print('Top models')
    Top_models = compare_models(exclude=None, include=None, fold=10, round=4, sort=metric,
                                n_select=top_n, budget_time=0, turbo=turbo, verbose=False)

    print('Fine tuning')
    Tuned_models = [tune_model(model, optimize=metric, n_iter=grid, verbose=False)
                    for model in Top_models]
    if ensemble:
        print('Blending')
        blend_model = blend_models(estimator_list='All', fold=10, round=4,
                                   choose_better=True, optimize=metric, turbo=turbo, verbose=False)
        print('Stacking')
        stacked_model = stack_models(estimator_list=Tuned_models, meta_model=None, fold=10, round=4,
                                     restack=True, choose_better=True, optimize=metric, verbose=False)

    print('Finalizing')
    best = automl(optimize=metric, use_holdout=True)
    final = finalize_model(best)
    save_model(final, model_name, model_only=False, verbose=False)

    if test_data is not None:
        estimate = predict_model(final, test_data)
        return final, get_logs(save=True), estimate
    return final, get_logs(save=True)
