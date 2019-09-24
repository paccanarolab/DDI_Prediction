from keras import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from keras import regularizers
from keras.layers import Activation, Dropout
from keras.initializers import RandomNormal
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

import tensorflow as tf  
from keras.backend.tensorflow_backend import set_session  



DATA_DIR = 'signatures_merging/'
cv_data_dir = 'ddi_split_perc_drugs/'
h_df = pkl.load(open(DATA_DIR+'h_df.p','rb'))
H = h_df.values
h_weights = [H.T]

x_train = pkl.load(open(cv_data_dir+'training_set_features_S_20.p','rb'))
y_train = pkl.load(open(cv_data_dir+'training_set_labels_X_20.p','rb'))
x_test = pkl.load(open(cv_data_dir+'testing_set_features_S_20.p','rb'))
y_test = pkl.load(open(cv_data_dir+'testing_set_labels_X_20.p','rb'))
cv_splits = pkl.load(open(cv_data_dir+'cv_splits_20.p','rb'))

def custom_activation(x):
    return x
get_custom_objects().update({'custom_activation': Activation(custom_activation)})

epochs = 100
batch_size = 128

def create_model(neurons_lay2=50, neurons_extra_layer = 20, l2_reg=0.0, activ_l2='tanh', extra_layers=2, dropout=False):
    # Instantiate model
    model = Sequential()
    # Layer 2
    model.add(Dense(neurons_lay2, activation=activ_l2, kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(l2_reg), input_shape=(20,)))

    for i in range(extra_layers):
        if dropout:
            model.add(Dropout(0.2))
        model.add(Dense(neurons_extra_layer, activation='relu', kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(l2_reg)))
    
    model.add(Dense(10, activation='sigmoid', kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(l2_reg)))
    # Fixed output layer
    fixed_layer = Dense(544,activation=custom_activation,use_bias=False)
    fixed_layer.trainable = False
    model.add(fixed_layer)
    model.layers[-1].set_weights(h_weights)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model

neurons_lay2=[20]
neurons_extra_layer=[20]
l2_reg=[0.0]
activ_l2=['tanh','sigmoid']
extra_layers=[5,15]
dropout = [False]

grid_params = list(itertools.product(neurons_lay2,neurons_extra_layer,l2_reg,activ_l2,extra_layers,dropout))
auc_scores = []
aupr_scores = []

print('Cross validation with {} different combinations of parameters.\n'.format(len(grid_params)))


config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
# config.log_device_placement = True  # to log device placement (on which device the operation ran)  
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)  
set_session(sess)  # set this TensorFlow session as the default session for Keras  


neurl2 = 0
neur_extr = 1
l2 = 2
actL2 = 3
extra = 4
drop = 5
auc_scores_grid = []
aupr_scores_grid = []
for i, param_comb in enumerate(grid_params):
    auc_scores = []
    aupr_scores = []
    model = create_model(param_comb[neurl2],param_comb[neur_extr],param_comb[l2], param_comb[actL2],param_comb[extra],param_comb[drop])
    print('Model created: Neur_lay2: {}, Ext_lay_neur: {}, L2_reg: {}, Activ_lay2: {}, Number_ext_lay: {}, Dropout: {}.\n'.format(param_comb[neurl2],param_comb[neur_extr],param_comb[l2], param_comb[actL2],param_comb[extra],param_comb[drop]))
    for j, fold in enumerate(cv_splits):
        print('Fold {}\n'.format(j+1))
        x_train_cv = fold['training_set_features_S']
        y_train_cv = fold['training_set_labels_X']
        x_valid_cv = fold['test_set_features_S']
        y_valid_cv= fold['test_set_labels_X']
        history = model.fit(x_train_cv,y_train_cv,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
        predicted = model.predict(x_valid_cv)
        auc_scores.append(roc_auc_score(y_valid_cv.flatten(),predicted.flatten()))
        aupr_scores.append(average_precision_score(y_valid_cv.flatten(),predicted.flatten()))

    auc_scores_grid.append(sum(auc_scores)/len(auc_scores))
    aupr_scores_grid.append(sum(aupr_scores)/len(aupr_scores))
    print('Percentage done: {}% - Iterations done: {}/{}'.format(((i+1)/len(grid_params)*100),i+1,len(grid_params)))



print('\n\nBest AUPR: {}\n'.format(max(aupr_scores_grid)))
print('Best AUC: {}\n'.format(max(auc_scores_grid)))
print('Best parameters combination for AUC: {}\n'.format(grid_params[auc_scores_grid.index(max(auc_scores_grid))]))
print('Best parameters combination for AUPR: {}\n'.format(grid_params[aupr_scores_grid.index(max(aupr_scores_grid))]))

best_comb = grid_params[auc_scores_grid.index(max(auc_scores_grid))]


model = create_model(best_comb[neurl2],best_comb[neur_extr],best_comb[l2], best_comb[actL2],best_comb[extra],best_comb[drop])
# model = create_model(25, 50, 5.0, 'tanh', 3, True)
# model = create_model(120, 90, 0.0, 'tanh', 'relu', 'sigmoid', True)
history = model.fit(x_train,y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
predicted = model.predict(x_test)
print('\nAUROC TEST: {}\n'.format(roc_auc_score(y_test.flatten(),predicted.flatten())))
print('AUPR TEST: {}\n'.format(average_precision_score(y_test.flatten(),predicted.flatten())))