import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

def find_best_weight(preds, target):
    def _validate_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, preds):
            final_prediction += weight * prediction
        return np.sqrt(mean_squared_error(final_prediction, target))

    # the algorithms need a starting value, right not we chose 0.5 for all weights
    # its better to choose many random starting points and run minimize a few times
    starting_values = [0.5] * len(preds)

    # adding constraints and a different solver as suggested by user 16universe
    # https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    # our weights are bound between 0 and 1
    bounds = [(0, 1)] * len(preds)

    res = minimize(_validate_func, starting_values, method='Nelder-Mead', bounds=bounds, constraints=cons)

    print('Ensemble Score: {best_score}'.format(best_score=(1 - res['fun'])))
    print('Best Weights: {weights}'.format(weights=res['x']))

    return res

# res = find_best_weight([oof_lgb, oof_xgb], target)