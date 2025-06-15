import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#%% Functions
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('train_history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show() 
    
def chi2(probe,ML):
    n=np.size(probe)
    sigma=0
    for i in range(n):
        sigma=sigma+((probe[i]-ML[i])**2)/probe[i]
    return sigma

def plot_compare(y_test, y_pred, title,ytitle,ymin,ymax):
    time = np.linspace(0, len(y_test), num=len(y_test))
    
    fig = plt.figure(constrained_layout=True, figsize=(8,6))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    plt.plot(time, y_test, label='Probe', color='black', linewidth=2)
    plt.plot(time, y_pred, label='Prediction', color='red', linewidth=2)
    
    plt.title(title, fontweight='bold', fontsize=16) 
    plt.xlabel("Index", color='black', fontsize=14)
    # plt.ylabel("Concentration (ppb)", color='black', fontsize=12)
    plt.ylabel(ytitle, color='black', fontsize=14)
    plt.ylim(ymin, ymax)
    plt.legend(loc="upper right", fontsize=12)
    
    plt.show()
    
#%% Models    
def model_SVM(x_train, x_test, y_train_te, y_train_ne):
    Temodel=svm.SVR(kernel='rbf', C=110, gamma=0.1)
    Temodel.fit(x_train,y_train_te[:,0])
    y_predict_te=Temodel.predict(x_test)
    
    Nemodel=svm.SVR(kernel='rbf', C=110, gamma=0.1)
    Nemodel.fit(x_train,y_train_ne[:,0])
    y_predict_ne=Nemodel.predict(x_test)    
    
    return y_predict_te, y_predict_ne

def model_XGBoost(x_train, x_test, y_train_te, y_train_ne):
    params_te = {
        'n_estimators': 1000,
        "max_depth": 6,  # Maximum depth of each tree
        "learning_rate": 0.3,  # Step size shrinkage to prevent overfitting
        "subsample": 0.35,
        "colsample_bytree" :0.7,
    }
    
    xgb_reg_te = xgb.XGBRegressor(**params_te)
    xgb_reg_te.fit(x_train, y_train_te)
    y_predict_te = xgb_reg_te.predict(x_test)
    
    dtrain_ne = xgb.DMatrix(x_train, label=y_train_ne)
    params_ne = {
        "eval_metric": "rmse",  # Root Mean Squared Error
        "max_depth": 19,  # Maximum depth of each tree
        "learning_rate": 0.05,  # Step size shrinkage to prevent overfitting
        "subsample": 0.35,
        "colsample_bytree" :0.8,
    }
    model_ne = xgb.train(params_ne, dtrain_ne)
    dtest = xgb.DMatrix(x_test)
    y_predict_ne = model_ne.predict(dtest)    
    
    return y_predict_te, y_predict_ne

#%% Load data
df=pd.read_csv("A-F_z=50mm_HeI_i-ratios.tsv", sep='\t')

x_data = df[["667.8/728.1", "728.1/706.5", "(501.6+504.8)/728.1","492.2/728.1","492.2/471.3","492.2/447.1"]].to_numpy()
y_data_ne= df[["probe-ne"]].to_numpy()
y_data_te= df[["probe-Te"]].to_numpy()
#%% Data splitting

cond_A = np.arange(0,57)
cond_B = np.arange(57,114)
cond_C = np.arange(114,171)
cond_D = np.arange(171,228)
cond_E = np.arange(228,285)
cond_F = np.arange(285,342)
cond_all = np.arange(0,342)
cond_Rand = np.array(random.sample(list(cond_all),round(np.size(cond_all)*0.2)))

testing = cond_F #change this for different testing data, the rest will be training

training=np.array([x for x in cond_all if x not in testing])

x_train = x_data[training] 
y_train_te = y_data_te[training]
y_train_ne = y_data_ne[training]
x_test = x_data[testing]
y_test_te = y_data_te[testing]
y_test_ne = y_data_ne[testing]

#%% model predictions

#y_predict_te, y_predict_ne = model_SVM(x_train, x_test, y_train_te, y_train_ne)
y_predict_te, y_predict_ne = model_XGBoost(x_train, x_test, y_train_te, y_train_ne)

#%% Plot results

X2=chi2(y_test_te[:,0], y_predict_te)
R2=r2_score(y_test_te[:,0], y_predict_te)
print("Te X2",X2)
print("Te R2",R2)
plot_compare(y_test_te[:,0], y_predict_te,  "electron temperature X2: "+str(round(X2,2))+" R2: "+str(round(R2,2)),"Te",0,10)
plt.figure(figsize=(6,6))
plt.scatter(y_test_te[:,0], y_predict_te,s=10)
plt.axline((0, 0), slope=1,color='r')
plt.xlabel("Probe", color='black', fontsize=12)
plt.ylabel("Predicted", color='black', fontsize=12)
plt.title("electron temperature X2: "+str(round(X2,2))+" R2: "+str(round(R2,2)), fontweight='bold', fontsize=14) 

#plt.xlim([3,7])
#plt.ylim([3,7])
plt.xlim([0,10])
plt.ylim([0,10])

X2=chi2(y_test_ne[:,0], y_predict_ne)
R2=r2_score(y_test_ne[:,0], y_predict_ne)
print("Ne X2",X2)
print("Ne R2",R2)
plot_compare(y_test_ne[:,0], y_predict_ne,  "electron density X2: "+str(round(X2,2))+" R2: "+str(round(R2,2)),"Ne",0,4)
plt.figure(figsize=(6,6))
plt.scatter(y_test_ne[:,0], y_predict_ne,s=10)
plt.axline((0, 0), slope=1,color='r')
plt.xlabel("Probe", color='black', fontsize=12)
plt.ylabel("Predicted", color='black', fontsize=12)
plt.title("electron density X2: "+str(round(X2,2))+" R2: "+str(round(R2,2)), fontweight='bold', fontsize=14) 

plt.xlim([0,4])
plt.ylim([0,4])
# plt.xlim([0.4,0.75])
# plt.ylim([0.4,0.75])
