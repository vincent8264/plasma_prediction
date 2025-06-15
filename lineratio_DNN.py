import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from keras.utils import plot_model
from keras.models import Model, Sequential
import keras.layers as layers
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras.layers import Input, Dropout, LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend import clear_session
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

#%%
def chi2(probe,ML):
    n=np.size(probe)
    sigma=0
    for i in range(n):
        sigma=sigma+((probe[i]-ML[i])**2)/probe[i]
    return sigma
def ANN(train_inputs, train_labels, filepath, num_epoch):
    num_feature = np.size(train_inputs,1)
    
    inputs = Input(shape=(num_feature))
    x = Dense(units=32,activation='relu')(inputs)
    x = Dense(units=32,activation='relu')(x)
    
    x = Dense(units=1,activation='linear')(x)
    model=Model(inputs, x)
    model.summary()
    
    model.compile(optimizer='Adam', loss="mae", metrics=['mae'])
    savebestmodel = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    
    train_history = model.fit(train_inputs,train_labels,
                              epochs=num_epoch, callbacks=[earlystop, savebestmodel],
                              batch_size=50, validation_split=0.2)
    return train_history

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('train_history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()   
    
def plot_compare(y_test, y_pred, title,ytitle,ymin,ymax):
    time = np.linspace(0, len(y_test), num=len(y_test))
    
    fig = plt.figure(constrained_layout=True, figsize=(8,6))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    plt.plot(time, y_test, label='Probe', color='black', linewidth=2)
    plt.plot(time, y_pred, label='SVM Prediction', color='red', linewidth=2)
    
    plt.title(title, fontweight='bold', fontsize=16) 
    plt.xlabel("Index", color='black', fontsize=14)
    # plt.ylabel("Concentration (ppb)", color='black', fontsize=12)
    plt.ylabel(ytitle, color='black', fontsize=14)
    plt.ylim(ymin, ymax)
    plt.legend(loc="upper right", fontsize=12)
    
    plt.show()
#%%read data
df=pd.read_csv("A-F_z=50mm_HeI_i-ratios.tsv", sep='\t')

x_data = df[["667.8/728.1", "728.1/706.5", "(501.6+504.8)/728.1","492.2/728.1","492.2/471.3","492.2/447.1"]].to_numpy()

# predict density or temperature
y_data= df[["probe-ne"]].to_numpy()
#y_data= df[["probe-Te"]].to_numpy()
#%%
# Normalization
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

cond_A=np.arange(0,57)
cond_B=np.arange(57,114)
cond_C=np.arange(114,171)
cond_D=np.arange(171,228)
cond_E=np.arange(228,285)
cond_F=np.arange(285,342)
cond_all=np.arange(0,342)
cond_Rand=np.array(random.sample(list(cond_all),round(np.size(cond_all)*0.2)))

testing=cond_A #change this for different testing data, the rest will be training

training=np.array([x for x in cond_all if x not in testing])

x_train = x_data[training]
y_train = y_data[training]
x_test = x_data[testing]
y_test = y_data[testing]

#%%
num_epoch = 50
modelpath = "C:/Users/User/Documents/models"

train_history = ANN(x_train, y_train, modelpath, num_epoch)

show_train_history(train_history, "loss", "val_loss")

model = load_model(modelpath)
y_predict = model.predict(x_test)

#%%
X2=chi2(y_test[:,0], y_predict[:,0])
R2=r2_score(y_test[:,0], y_predict[:,0])
print("Ne X2",X2)
print("Ne R2",R2)
plot_compare(y_test[:,0], y_predict, "electron density X2: "+str(round(X2,2))+" R2: "+str(round(R2,2)),"Te",0,10)
plt.figure(figsize=(6,6))
plt.scatter(y_test[:,0], y_predict[:,0],s=10)
plt.axline((0, 0), slope=1,color='r')
plt.xlabel("Probe", color='black', fontsize=12)
plt.ylabel("Predicted", color='black', fontsize=12)
plt.title("electron density X2: "+str(round(X2,2))+" R2: "+str(round(R2,2)), fontweight='bold', fontsize=14) 

#plt.xlim([3,7])
#plt.ylim([3,7])
plt.xlim([0,4])
plt.ylim([0,4])