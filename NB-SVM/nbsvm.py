from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.externals import joblib

def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(x, y):
    y = y.values
    r = np.log(pr(x,1,y) / pr(x,0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

# fit model and make predictions
# We have 6 classes, so we do a single class NB-SVM on each class
def train_model(train, train_x, labels, write_filename=None): 
    model_weights_dict = {}   
    for i, j in enumerate(labels):
        print('fitting', j)
        model, weights = get_mdl(train_x, train[j])
        # store model and weights value for future predictions
        model_weights_dict[i] = weights, model
    if write_filename is not None:
        joblib.dump(model_weights_dict, write_filename, compress=1)
    return model_weights_dict
   

def get_preds_from_model(model, test_x, labels):
    preds = np.zeros((test_x.shape[0], len(labels)))
    for i,j in enumerate(labels):
        weights, m = model.values()[i]
        preds[:,i] = m.predict_proba(test_x.multiply(weights))[:,1]
    return preds

