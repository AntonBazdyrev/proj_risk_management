from task2.linreg import fit_predict as linreg_fit_predict
from task2.elasticnet import fit_predict as elasticnet_fit_predict
from task2.lgbm_gbdt import fit_predict as lgbm_gbdt_fit_predict
from task2.lstm import fit_predict as lstm_fit_predict
from task2.transformer import fit_predict as transformer_fit_predict

get_model_fit_predict = {
    'lin_reg': linreg_fit_predict, 
    'elastic_net': elasticnet_fit_predict, 
    'lgbm': lgbm_gbdt_fit_predict, 
    'lstm': lstm_fit_predict, 
    'transformer': transformer_fit_predict
}