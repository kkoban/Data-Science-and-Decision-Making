import numpy as np

def get_ps_weights(model, x, t): # to compute sample weights
  ti = np.squeeze(t)
  model.fit(x, ti)
  ptx = model.predict_proba(x).T[1].T + 0.0001 
  wi = ti/ptx + 1 - ti/ 1 - ptx
  return wi


def feat_imp(model): # to find feature importance for fitted model

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    return importances, indices




# Standard Effect Metrices for Causal Effect Estimation

def pehe(effect_true, effect_pred): # to calculate error for data with effects/counterfactuals
    return np.sqrt(np.mean((effect_pred - effect_true)**2))



def policy_risk(effect_pred, yf, t, e): # to calculate Policy Risk for data mixed with RCTs

    t_e = t[e > 0]
    yf_e = yf[e > 0]
    effect_pred_e = effect_pred[e > 0]

    if np.any(np.isnan(effect_pred_e)):
        return np.nan

    policy = effect_pred_e > 0.0
    treat_overlap = (policy == t_e) * (t_e > 0)
    control_overlap = (policy == t_e) * (t_e < 1)

    if np.sum(treat_overlap) == 0:
        treat_value = 0
    else:
        treat_value = np.mean(yf_e[treat_overlap])

    if np.sum(control_overlap) == 0:
        control_value = 0
    else:
        control_value = np.mean(yf_e[control_overlap])

    pit = np.mean(policy)
    policy_value = pit * treat_value + (1.0 - pit) * control_value

    return 1.0 - policy_value


