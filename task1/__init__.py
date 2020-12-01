from task1.logreg import fit_predict as logreg_fit_predict
from task1.elasticnet import fit_predict as elasticnet_fit_predict
from task1.lgbm_gbdt import fit_predict as lgbm_gbdt_fit_predict
from task1.lstm import fit_predict as lstm_fit_predict
from task1.transformer import fit_predict as transformer_fit_predict

get_model_fit_predict = {
    'log_reg': logreg_fit_predict, 
    'elastic_net': elasticnet_fit_predict, 
    'lgbm': lgbm_gbdt_fit_predict, 
    'lstm': lstm_fit_predict, 
    'transformer': transformer_fit_predict
}