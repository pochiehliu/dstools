from pycaret.classification import *


def clas(data, target, metric='Accuracy', session_id=42, model_name='model',
         turbo=True, top_n=5, use_gpu=False, n_iter=10, test_data=None,
         fix_imbalance=False, fix_imbalance_method=None):
    'Apply pycaret for fast classification model building'

    print('Setup')
    cls_exp = setup(data=data, target=target, session_id=session_id,
                    fix_imbalance=fix_imbalance, fix_imbalance_method=fix_imbalance_method,
                    silent=True, html=False, verbose=False, use_gpu=use_gpu,
                    log_experiment=True, log_plots=True, log_profile=True, log_data=True)

    '''
    sort option: 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'
    '''
    print('Top models')
    Top_models = compare_models(exclude=None, include=None, fold=10, round=4, sort=metric,
                                n_select=top_n, budget_time=0, turbo=turbo, verbose=False)

    '''
    fine tune top n models
    n_iter: Number of iterations within the Random Grid Search.
    '''
    print('Fine tuning')
    Tuned_models = [tune_model(model, optimize=metric, n_iter=n_iter, verbose=False)
                    for model in Top_models]

    '''
    blending and staking
    optimize option: : 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'
    '''
    print('Blending')
    blend_model = blend_models(estimator_list='All', fold=10, round=4,
                               choose_better=True, optimize=metric, turbo=turbo, verbose=False)
    print('Stacking')
    stacked_model = stack_models(estimator_list=Tuned_models, meta_model=None, fold=10, round=4,
                                 restack=True, choose_better=True, optimize=metric, verbose=False)

    '''
    save the finalized best model based on holdout set metric
    '''
    print('Finalizing')
    best = automl(optimize=metric, use_holdout=True)
    final = finalize_model(best)
    save_model(final, model_name, model_only=False, verbose=False)

    if test_data is not None:
        estimate = predict_model(final, test_data)
        return final, get_logs(save=True), estimate
    return final, get_logs(save=True)
