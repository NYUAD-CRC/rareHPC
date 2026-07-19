import time
import pandas as pd
import seaborn as sns
from sklearn.base import clone
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import sys, joblib, os
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from category_encoders.binary import BinaryEncoder as bec
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import Lasso
from category_encoders.binary import BinaryEncoder as bencoder
import argparse

# parameters for Hyper-parameter tuning for different models
param_grids = {
        'Linear Regression': {},

        'Ridge': {
            # default alpha=1.0
            'alpha': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        },

        'Lasso': {
            # default alpha=1.0
            'alpha': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        },

        'Random Forest': {
            # defaults: n_estimators=100, max_depth=None,
            # min_samples_split=2, min_samples_leaf=1, max_features=1.0
            'n_estimators': [80, 100, 150, 200, 500, 1000],
            'max_depth': [None, 10, 15, 20],
            'min_samples_split': [2, 3, 4, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_features': [1.0, 'sqrt', 'log2']
        },

        'XGBoost': {
            # common defaults: n_estimators=100, max_depth=6,
            # learning_rate=0.3, subsample=1, colsample_bytree=1
            'n_estimators': [80, 100, 150, 200, 500, 1000],
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.1, 0.2, 0.3, 0.4],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3],
            'reg_alpha': [0, 0.001, 0.01, 0.1],
            'reg_lambda': [0.5, 1, 1.5, 2]
        },

        'LGBM': {
            # common defaults: n_estimators=100, learning_rate=0.1,
            # num_leaves=31, max_depth=-1
            'n_estimators': [80, 100, 150, 200, 500, 1000],
            'learning_rate': [0.05, 0.08, 0.1, 0.12, 0.15],
            'max_depth': [-1, 5, 7, 10],
            'num_leaves': [15, 31, 45, 63],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.001, 0.01, 0.1],
            'reg_lambda': [0, 0.1, 1.0, 2.0]
        }
    }


def bucketing(x):
    if x<=5:
        return '0-5'
    elif x<=10:
        return '6-10'
    elif x<=20:
        return '11-20'
    elif x<=50:
        return '21-50'
    else:
        return '>50'

def RandomForestClasification(Xtrain, ytrain, Xtest, ytest, n_iter=20):
    model_name = 'Random Forest'
    best_recall = -np.inf
    best_precision = -np.inf
    best_f1 = -np.inf
    best_accuracy = -np.inf
    best_params = {}
    tuned_models = {}
    for index, params in enumerate(ParameterSampler(param_grids[model_name], n_iter=n_iter, random_state=42)):
        model = RandomForestClassifier(random_state=42, n_jobs=18, class_weight='balanced')
        if index<n_iter-1:
            model.set_params(**params)
        else:
            print('Default Params')
        model.fit(Xtrain,ytrain)
        ypred = model.predict(Xtest)
        recall = recall_score(ytest, ypred, pos_label='High')
        precision = precision_score(ytest, ypred, pos_label='High')
        f1 = f1_score(ytest, ypred, pos_label='High')
        accuracy = accuracy_score(ytest, ypred)
        if recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_f1 = f1
            best_accuracy = accuracy
            best_params[model_name] = params
            tuned_models[model_name] = model
    print('Best Params for', model_name, best_params[model_name])
    print('Best Recall {} Precision {} F1 {} Accuracy {} for {}'.format(best_recall, best_precision, best_f1, best_accuracy, model_name))
    return round(best_accuracy,2), round(best_recall,2), round(best_precision,2), tuned_models[model_name], best_params[model_name]

def XGBoostClasification(Xtrain, ytrain, Xtest, ytest, n_iter=20):
    model_name = 'XGBoost'
    best_recall = -np.inf
    best_precision = -np.inf
    best_f1 = -np.inf
    best_accuracy = -np.inf
    best_params = {}
    tuned_models = {}
    for index, params in enumerate(ParameterSampler(param_grids[model_name], n_iter=n_iter, random_state=42)):
        model = XGBClassifier(random_state=42, n_jobs=18, class_weight='balanced')
        if index<n_iter-1:
            model.set_params(**params)
        else:
            print('Default Params')
        model.fit(Xtrain,ytrain)
        ypred = model.predict(Xtest)
        recall = recall_score(ytest, ypred, pos_label='High')
        precision = precision_score(ytest, ypred, pos_label='High')
        f1 = f1_score(ytest, ypred, pos_label='High')
        accuracy = accuracy_score(ytest, ypred)
        if recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_f1 = f1
            best_accuracy = accuracy
            best_params[model_name] = params
            tuned_models[model_name] = model
    print('Best Params for', model_name, best_params[model_name])
    print('Best Recall {} Precision {} F1 {} Accuracy {} for {}'.format(best_recall, best_precision, best_f1, best_accuracy, model_name))
    return round(best_accuracy,2), round(best_recall,2), round(best_precision,2), tuned_models[model_name], best_params[model_name]

def lgbmClasification(Xtrain, ytrain, Xtest, ytest, n_iter=20):
    model_name = 'LGBM'
    best_recall = -np.inf
    best_precision = -np.inf
    best_f1 = -np.inf
    best_accuracy = -np.inf
    best_params = {}
    tuned_models = {}
    for index, params in enumerate(ParameterSampler(param_grids[model_name], n_iter=n_iter, random_state=42)):
        model = lightgbm.LGBMClassifier(random_state=42, n_jobs=18, class_weight='balanced', verbose=-1)
        if index<n_iter-1:
            model.set_params(**params)
        else:
            print('Default Params')
        model.fit(Xtrain,ytrain)
        ypred = model.predict(Xtest)
        recall = recall_score(ytest, ypred, pos_label='High')
        precision = precision_score(ytest, ypred, pos_label='High')
        f1 = f1_score(ytest, ypred, pos_label='High')
        accuracy = accuracy_score(ytest, ypred)
        if recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_f1 = f1
            best_accuracy = accuracy
            best_params[model_name] = params
            tuned_models[model_name] = model
    print('Best Params for', model_name, best_params[model_name])
    print('Best Recall {} Precision {} F1 {} Accuracy {} for {}'.format(best_recall, best_precision, best_f1, best_accuracy, model_name))
    return round(best_accuracy,2), round(best_recall,2), round(best_precision,2), tuned_models[model_name], best_params[model_name]

def classification_model(df_modelling, dump_path):
    
    """
    Trains a classification model to predict whether the compute time utilized by a job is 'High' or 'Low' based on various features in the dataset.
    """
    
    dump_path = os.path.join(os.path.dirname(dump_path), 'classification_model')
    os.makedirs(dump_path, exist_ok=True)
    df_modelling = df_modelling.sort_values(by = ['time_submit', 'id_user', 'id_job'])
    df_model_sel = df_modelling[['id_job', 'id_user', 'id_assoc', 'id_qos', 'id_group', 'constraints', 'partition', 'account', 'job_name', 'wait_time', 
                             'wall_time_hrs', 'allocated_memory_gb', 'time_submit', 'cpus_req', 'gpu_req', 'compute_utilised_time_hrs', 'used_memory_gb', 
                             'last_memuse_gb', 'last_timeuse_hrs', 'rollingmean_3mem', 'rollingmean_3time', 'rollingmean_7mem', 'rollingmean_7time', 
                             'rollingmean_10mem', 'rollingmean_10time','txt_based_mem', 'txt_based_time', 'job_array_idx', 'last_memuse_gb%', 'last_timeuse_hrs%', 
                             'mem_rollingmean%', 'time_rollingmean%', 'rollingmean_3mem%', 'rollingmean_3time%', 'rollingmean_7mem%', 
                             'rollingmean_7time%', 'rollingmean_10mem%', 'rollingmean_10time%']]
    
    df_model_sel['computeTime_bucket'] = df_model_sel['compute_utilised_time_hrs'].apply(lambda x:'High' if x>(1/60) else 'Low')
    bec_path  = os.path.join(dump_path, 'encoders', 'ClassFier_bec_alldata'+'.joblib')
    model_path = os.path.join(dump_path, 'models', 'ClassFier_LGBM_alldata'+'.joblib')
    os.makedirs(os.path.dirname(bec_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    remove_cols = ['job_name', 'compute_utilised_time_hrs', 'used_memory_gb', 'computeTime_bucket', 'constraints', 'partition', 'id_group', 'id_qos','id_job']
    keep_cols = [col for col in df_model_sel.columns if col not in remove_cols]

    df_model_sel.sort_values(by=['time_submit', 'id_user', 'id_job'], inplace=True)
    y_all = df_model_sel[['compute_utilised_time_hrs','used_memory_gb', 'computeTime_bucket']]
    
    
    df_train_all = df_model_sel[keep_cols]
    total = df_train_all.shape[0]
    train_idx, test_idx = train_test_split(range(0,total), test_size=0.2, shuffle=False)
    df_train, df_test = df_train_all.iloc[train_idx], df_train_all.iloc[test_idx]
    y_train, y_test = y_all['computeTime_bucket'].iloc[train_idx], y_all['computeTime_bucket'].iloc[test_idx]
    
    categ_cols = ['id_user', 'account', 'id_assoc']
    
    be = bec(cols=categ_cols)
    df_res_train = be.fit_transform(df_train)
    Xtrain, ytrain = df_res_train.values, y_train.values
    Xtest, ytest = be.transform(df_test).values, y_test.values
    accuracy, recall, precision, modellgbm, lgbest_params = lgbmClasification(Xtrain, ytrain, Xtest, ytest, n_iter=20)
    print(accuracy, recall, precision)
    # accuracy, recall, precision, modelrf, rfbest_params = RandomForestClasification(Xtrain, ytrain, Xtest, ytest, n_iter=100)
    # accuracy, recall, precision, modelxgb, xgbest_params = XGBoostClasification(Xtrain, ytrain, Xtest, ytest, n_iter=100)
    df_res_test = be.transform(X=df_test)
    pred_utilizCategory = modellgbm.predict(df_res_test.values)
    
    be = bec(cols=categ_cols)
    X_alltr, y_alltr = be.fit_transform(df_train_all).values, y_all['computeTime_bucket'].values
    modellgbm = lightgbm.LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=18)
    modellgbm.set_params(**lgbest_params)
    modellgbm.fit(X_alltr,y_alltr)
    joblib.dump(be, bec_path)
    joblib.dump(modellgbm, model_path)
    cfm = confusion_matrix(y_test.values, pred_utilizCategory, labels=['Low','High'])
    print(cfm)
    print(confusion_matrix(y_test.values, pred_utilizCategory, labels=['Low','High'], normalize='all'))
    cmap=sns.color_palette("light:b", as_cmap=True) 
    sns.heatmap(cfm.T, annot=True, cmap=cmap, fmt='d', xticklabels=['Low','High'], yticklabels=['Low','High'], cbar=False,annot_kws={"fontsize":15})
    plt.xlabel('GroundTruth', fontweight='bold')
    plt.ylabel('Model Predictions', fontweight='bold')
    plt.savefig(dump_path+'/Classifier_cfm.png')
    return df_modelling

def plot_clean2(df_plot, labelx, title, labely, save_path, topk1=15, topk2=10):
    
    """
    Helper function to generate a grouped bar plot for total savings and percentage savings based on the provided DataFrame.
    """
    
    df_plot[labely+'_total_text'] = df_plot[labely+'_total'].apply(lambda x:round(x,2))
    df_plot[labely+'_percent_text'] = df_plot[labely+'_percentage'].apply(lambda x:str(round(x,2))+' %')
    print(round(df_plot[labely+'_percentage'].mean(), 2), '% average savings for', labely)
    
    df_plot.sort_values(by=labely+'_total_text', inplace=True, ascending=False)
    df_plot= df_plot.iloc[:topk1]
    df_plot.sort_values(by=labely+'_percent_text', inplace=True, ascending=False)
    df_plot= df_plot.iloc[:topk2]
    df_plot.sort_values(by=labely+'_total_text', inplace=True, ascending=False)
    df_plot= df_plot.iloc[:topk2]
    
    nature_colors = ['#E64B35', '#4DBBD5']  # Red and Blue from nature style
    fig = go.Figure()
    if labely=='savings_memory_gb_1bucket' or labely=='improved_computetime_hrs_1bucket':
        
        fig.add_trace(go.Bar(name='<b>{}</b>'.format('Total_savings'), x=df_plot[labelx], y=df_plot[labely+'_total'], text=[f"<b>{x}</b>" for x in df_plot[labely+'_total_text']],
        marker_color=nature_colors[0], textfont=dict(size=15, family='Arial', color='black'), textposition='inside', yaxis="y1", showlegend=False, offsetgroup = 0))

        fig.add_trace(go.Bar(name='<b>{}</b>'.format('Percentage_savings'), x=df_plot[labelx], y=df_plot[labely+'_percentage'], text=[f"<b>{x}</b>" for x in df_plot[labely+'_percent_text']],
                marker_color=nature_colors[1], textfont=dict(size=15, family='Arial', color='black'), textposition='outside', showlegend=False, offsetgroup=1, yaxis='y2'))

        fig.add_annotation(text="(b)", xref="paper", yref="paper", x=-0.05, y=1.1, showarrow=False, font=dict(size=15, color="black", family="Arial, sans-serif"), align="left", valign="top")
        
    else:    
        
        fig.add_trace(go.Bar(name='<b>{}</b>'.format('Total_savings'), x=df_plot[labelx], y=df_plot[labely+'_total'], text=[f"<b>{x}</b>" for x in df_plot[labely+'_total_text']],
                marker_color=nature_colors[0], textfont=dict(size=15, family='Arial', color='black'), textposition='inside', yaxis="y1", offsetgroup=0))

        
        fig.add_trace(go.Bar(name='<b>{}</b>'.format('Percentage_savings'), x=df_plot[labelx], y=df_plot[labely+'_percentage'], text=[f"<b>{x}</b>" for x in df_plot[labely+'_percent_text']],
                marker_color=nature_colors[1], textfont=dict(size=15, family='Arial', color='black'), textposition='outside', offsetgroup=1, yaxis='y2'))

        fig.add_annotation(text="(a)", xref="paper", yref="paper", x=-0.05, y=1.1, showarrow=False, font=dict(size=15, color="black", family="Arial, sans-serif"), align="left", valign="top")
        
    if labely == '%_UtilisedTime':
        labely='Utilised_time_%'
    labely = labely[0].upper() + labely[1:]
    title = title[0].upper()+title[1:]
    
    fig.update_layout(
    barmode='group', 
    xaxis=dict(title=f"<b>{labelx[0].upper()+labelx[1:]}</b>", title_font=dict(size=15,family='Arial', color='black'), tickfont=dict(size=15, family='Arial', color='black'),
        ticktext=[f"<b>{tick}</b>" for tick in df_plot[labelx]], tickvals=list(range(len(df_plot[labelx]))),),
    yaxis=dict(title=f"<b>{labely.split('_1bucket')[0]}</b>", title_font=dict(size=15, family='Arial', color='black'), tickfont=dict(size=10, family='Arial', color='black'), type="log"),
    yaxis2=dict(title="<b>Percentage Savings</b>", title_font=dict(size=15, family='Arial', color='black'), tickfont=dict(size=10, family='Arial', color='black'), overlaying="y", side="right",
                type="linear"),
    legend=dict(font=dict(size=10, family='Arial', color='black')))

    fig.write_image(save_path, scale=5)
    return

def merge_total_percentage(df, labelx, title, labely, save_path):
    """
    Merges total and percentage savings data for plotting.
    """
    
    t1 = df.groupby(labelx)[labely].sum()
    t1 = pd.DataFrame(t1)
    t1[labelx]=t1.index.to_list()
    t1.reset_index(drop=True, inplace=True)
    t1[labely] = t1[labely].apply(lambda x:round(x))
    t1.sort_values(by=labely, inplace=True, ascending=False)
    t1.rename(columns={labely:labely+'_total'}, inplace=True)
    if labely=='savings_memory_gb' or labely=='savings_memory_gb_1bucket':
        labely_ = 'allocated_memory_gb'
    elif labely=='improved_computetime_hrs' or labely=='improved_computetime_hrs_1bucket':
        labely_ = 'wall_time_hrs'
    else:
        print('INVALID INPUT')
        raise Exception('ERROR')
    
    t2 = df.groupby(labelx)[labely_].sum()
    t2 = pd.DataFrame(t2)
    t2[labelx]=t2.index.to_list()
    t2[labely_] = t2[labely_].apply(lambda x:round(x))
    t2.reset_index(drop=True, inplace=True)
    t2.rename(columns={labely_:labely+'_percentage'}, inplace=True)
    merged_df = t1.merge(t2, on=labelx, how='inner')
    merged_df[labely+'_percentage'] = merged_df.apply(lambda x:round(x[labely+'_total']*100/x[labely+'_percentage']), axis=1)
    merged_df.sort_values(by=labely+'_percentage', inplace=True, ascending=False)
    print(merged_df.columns)
    
    plot_clean2(merged_df, labelx, title, labely, save_path)

def RandomForest(Xtrain, ytrain, Xtest, ytest, n_iter=50):
    model_name = 'Random Forest'
    best_rmse = np.inf
    best_mae = np.inf
    best_mape = np.inf
    best_params = {}
    tuned_models = {}
    for index, params in enumerate(ParameterSampler(param_grids[model_name], n_iter=n_iter, random_state=42)):
        model = RandomForestRegressor(random_state=42, n_jobs=18)
        if index<n_iter-1:
            model.set_params(**params)
        else:
            print('Default Params')
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)*100
        ytest = ytest*100
        mae, rmse, mape = mean_absolute_error(ytest, ypred), round(np.sqrt(mean_squared_error(ytest, ypred)),4), mean_absolute_percentage_error(ytest, ypred)
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            best_params[model_name] = params
            tuned_models[model_name] = model
    print('Best Params for', model_name, best_params[model_name])
    print('Best RMSE {} MAE {} MAPE {} for {}'.format(best_rmse, best_mae, best_mape, model_name))
    return round(best_mae,2), round(best_rmse,2), round(best_mape,2), ypred, tuned_models[model_name], best_params[model_name]

def XgBoostReg(Xtrain,ytrain, Xtest, ytest, n_iter=50):
    model_name = 'XGBoost'
    best_rmse = np.inf
    best_mae = np.inf
    best_mape = np.inf
    best_params = {}
    tuned_models = {}
    for index, params in enumerate(ParameterSampler(param_grids[model_name], n_iter=n_iter, random_state=42)):
        model = XGBRegressor(objective='reg:linear', random_state=42, n_jobs=18)
        if index<n_iter-1:
            model.set_params(**params)
        else:
            print('Default Params')
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)*100
        ytest = ytest*100
        mae, rmse, mape = mean_absolute_error(ytest, ypred), round(np.sqrt(mean_squared_error(ytest, ypred)),4), mean_absolute_percentage_error(ytest, ypred)
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            best_params[model_name] = params
            tuned_models[model_name] = model
    print('Best Params for', model_name, best_params[model_name])
    print('Best RMSE {} MAE {} MAPE {} for {}'.format(best_rmse, best_mae, best_mape, model_name))
    return round(best_mae,2), round(best_rmse,2), round(best_mape,2), ypred, tuned_models[model_name], best_params[model_name]

def LightGBM(Xtrain, ytrain, Xtest, ytest, n_iter=50):
    model_name = 'LGBM'
    best_rmse = np.inf
    best_mae = np.inf
    best_mape = np.inf
    best_params = {}
    tuned_models = {}
    for index, params in enumerate(ParameterSampler(param_grids[model_name], n_iter=n_iter, random_state=42)):
        model = lightgbm.LGBMRegressor(objective='regression', random_state=42, n_jobs=18, verbose=-1)
        if index<n_iter-1:
            model.set_params(**params)
        else:
            print('Default Params')
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)*100
        ytest = ytest*100
        mae, rmse, mape = mean_absolute_error(ytest, ypred), round(np.sqrt(mean_squared_error(ytest, ypred)),4), mean_absolute_percentage_error(ytest, ypred)
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            best_params[model_name] = params
            tuned_models[model_name] = model
    print('Best Params for', model_name, best_params[model_name])
    print('Best RMSE {} MAE {} MAPE {} for {}'.format(best_rmse, best_mae, best_mape, model_name))
    return round(best_mae,2), round(best_rmse,2), round(best_mape,2), ypred, tuned_models[model_name], best_params[model_name]

def LassoReg(Xtrain, ytrain, Xtest, ytest, n_iter=5):
    ss = StandardScaler()
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.transform(Xtest)
    model_name = 'Lasso'
    best_rmse = np.inf
    best_mae = np.inf
    best_mape = np.inf
    best_params = {}
    tuned_models = {}
    for index, params in enumerate(ParameterSampler(param_grids[model_name], n_iter=n_iter, random_state=42)):
        model = Lasso(random_state=42)
        if index<n_iter-1:
            model.set_params(**params)
        else:
            print('Default Params')
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)*100
        ytest = ytest*100
        mae, rmse, mape = mean_absolute_error(ytest, ypred), round(np.sqrt(mean_squared_error(ytest, ypred)),4), mean_absolute_percentage_error(ytest, ypred)
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            best_params[model_name] = params
            tuned_models[model_name] = model
    print('Best Params for', model_name, best_params[model_name])
    print('Best RMSE {} MAE {} MAPE {} for {}'.format(best_rmse, best_mae, best_mape, model_name))
    
    return round(best_mae,2), round(best_rmse,2), round(best_mape,2), ypred, tuned_models[model_name], best_params[model_name]

def LinearReg(Xtrain, ytrain, Xtest, ytest, log_transform=False):
    ss = StandardScaler()
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.transform(Xtest)
    if log_transform:
        ytrain=np.log10(ytrain)
    lr = LinearRegression().fit(Xtrain, ytrain)
    ypred = lr.predict(Xtest)
    if log_transform:
        ypred = 10**ypred
    ypred = 100*ypred
    ytest = 100*ytest
    mae, rmse, mape = mean_absolute_error(ytest, ypred), round(np.sqrt(mean_squared_error(ytest, ypred)),4), mean_absolute_percentage_error(ytest, ypred)
    return round(mae,2), round(rmse,2), round(mape,2), ypred, lr

def data_balancing(df_train, y, ybucket, combined=False):

    """
    Balances the training dataset based on the target variable's distribution using SMOTENC and make_imbalance techniques. 
    """
    
    if y=='%_UtilisedTime':
        if combined:
            print('Before balancing', Counter(df_train[ybucket]))
            df_train_balanced, yb = make_imbalance(df_train, df_train[ybucket].to_list(), 
                                          sampling_strategy={'0-5': 100000, '6-10': 30334, '11-20': 26121, '21-50':90000, '>50':10726}, random_state=42)
        else:
            print('Before balancing', Counter(df_train[ybucket]))
            # df_train_balanced, yb = make_imbalance(df_train, df_train[ybucket].to_list(), 
            #                               sampling_strategy={'0-5': 100000, '6-10': 26100, '11-20': 22434, '21-50':90000, '>50':9218}, random_state=42)
            df_train_balanced, yb = make_imbalance(df_train, df_train[ybucket].to_list(), 
                                          sampling_strategy={'0-5': 100000, '6-10': 25784, '11-20': 22203, '21-50':90000, '>50':9116}, random_state=42)
        X = df_train_balanced[(df_train_balanced[ybucket]=='>50') | (df_train_balanced[ybucket]=='21-50')].drop(columns=['id_job', 'job_name', 'ComputeUtil_levels']+[ybucket])
        sm = SMOTENC(random_state=42, categorical_features=['id_user', 'id_assoc', 'id_qos', 'id_group', 'partition', 'account', 'constraints'])
        X_res, y_res = sm.fit_resample(X, df_train_balanced[(df_train_balanced[ybucket]=='>50') | (df_train_balanced[ybucket]=='21-50')][ybucket])
        df_train_smote = pd.concat([X_res[y_res=='>50'], df_train_balanced[df_train_balanced[ybucket]!='>50'][X_res.columns]], ignore_index=True)
        df_train_smote[ybucket]=df_train_smote[y].apply(lambda x:bucketing(x))
        print('After balancing', Counter(df_train_smote[ybucket]))
        
        X = df_train_smote[(df_train_smote[ybucket]=='6-10') | (df_train_smote[ybucket]=='21-50')].drop(columns=[ybucket])
        sm = SMOTENC(random_state=42, categorical_features=['id_user', 'id_assoc', 'id_qos', 'id_group', 'partition', 'account', 'constraints'])
        X_res, y_res = sm.fit_resample(X, df_train_smote[(df_train_smote[ybucket]=='6-10') | (df_train_smote[ybucket]=='21-50')][ybucket])
        df_train_smote = pd.concat([X_res[y_res=='6-10'], df_train_smote[df_train_smote[ybucket]!='6-10'][X_res.columns]], ignore_index=True)
        df_train_smote[ybucket]=df_train_smote[y].apply(lambda x:bucketing(x))
        print('After balancing', Counter(df_train_smote[ybucket]))
        
        X = df_train_smote[(df_train_smote[ybucket]=='11-20') | (df_train_smote[ybucket]=='21-50')].drop(columns=[ybucket])
        sm = SMOTENC(random_state=42, categorical_features=['id_user', 'id_assoc', 'id_qos', 'id_group', 'partition', 'account', 'constraints'])
        X_res, y_res = sm.fit_resample(X, df_train_smote[(df_train_smote[ybucket]=='11-20') | (df_train_smote[ybucket]=='21-50')][ybucket])
        df_train_smote = pd.concat([X_res[y_res=='11-20'], df_train_smote[df_train_smote[ybucket]!='11-20'][X_res.columns]], ignore_index=True)
        df_train_smote[ybucket]=df_train_smote[y].apply(lambda x:bucketing(x))
        
    else:
        if combined:
            print('Before balancing', Counter(df_train[ybucket]))
            df_train_balanced, yb = make_imbalance(df_train, df_train[ybucket].to_list(), 
                                            sampling_strategy={'0-5': 90000, '6-10': 72407, '11-20': 86478, '21-50':96401, '>50':23432}, random_state=42)
        else:
            print('Before balancing', Counter(df_train[ybucket]))
            try:
                df_train_balanced, yb = make_imbalance(df_train, df_train[ybucket].to_list(), 
                                            sampling_strategy={'0-5': 75000, '6-10': 61945, '11-20': 76411, '21-50':82434, '>50':20055}, random_state=42)
            except:
                try:
                    df_train_balanced, yb = make_imbalance(df_train, df_train[ybucket].to_list(), 
                                            sampling_strategy={'0-5': 75000, '6-10': 61945, '11-20': 76410, '21-50':82434, '>50':20055}, random_state=42)
                except:
                    try:
                        df_train_balanced, yb = make_imbalance(df_train, df_train[ybucket].to_list(), 
                                            sampling_strategy={'0-5': 75000, '6-10': 61546, '11-20': 73506, '21-50':81941, '>50':19917}, random_state=42)
                    except:
                        raise Exception('Invalid undersmapling entries')
                        
    
        X = df_train_balanced[(df_train_balanced[ybucket]=='>50') | (df_train_balanced[ybucket]=='6-10')].drop(columns=['id_job', 'job_name', 'ComputeUtil_levels']+[ybucket])
        sm = SMOTENC(random_state=42, categorical_features=['id_user', 'id_assoc', 'id_qos', 'id_group', 'partition', 'account', 'constraints'])
        X_res, y_res = sm.fit_resample(X, df_train_balanced[(df_train_balanced[ybucket]=='>50') | (df_train_balanced[ybucket]=='6-10')][ybucket])
        df_train_smote = pd.concat([X_res[y_res=='>50'], df_train_balanced[df_train_balanced[ybucket]!='>50'][X_res.columns]], ignore_index=True)
       
    print('After balancing', Counter(list(map(bucketing,df_train_smote[y]))))
    return df_train_smote

def get_pr(df, y, path):
    """
    Generates a grouped bar plot for precision, recall, and F1-score based on the provided DataFrame.
    """
    
    if y=='memory_efficiency_%':
        suffix='memory'
    else:
        suffix='computetime'
    # y = '%_UtilisedTime''
    categories = ['0-5', '6-10', '11-20', '21-50', '>50']
    prec = precision_score(df['actual_bucket_'+suffix], df['predicted_bucket_'+suffix], labels = categories, average=None)
    rec = recall_score(df['actual_bucket_'+suffix], df['predicted_bucket_'+suffix], labels = categories, average=None)
    fscore = f1_score(df['actual_bucket_'+suffix], df['predicted_bucket_'+suffix], labels = categories, average=None)
    rec = [round(item*100,2) for item in rec]
    prec = [round(item*100,2) for item in prec]
    fscore = [round(item*100,2) for item in fscore]
    print(prec, rec, fscore)
    df_temp = pd.DataFrame(columns = ['score_%', 'type', 'categories'])
    df_temp['Categories']= categories + categories+categories
    df_temp['Score_%'] = prec+rec+fscore
    df_temp['Metric'] = ['Precision' for item in prec]+['Recall' for item in rec] + ['F1score' for item in fscore]
    labely = 'Score_%'
    labelx = 'Categories'
    fig = px.bar(df_temp, x='Categories', y='Score_%', color='Metric', barmode="group", width =1200, height=400,text= list(map(lambda x:'<b>{}</b>'.format(x),df_temp[labely])))
    
    fig.update_layout(barmode='group', xaxis_title = '<b>{}</b>'.format(labelx), yaxis_title = '<b>{}</b>'.format(labely),
                      yaxis=dict(title_font=dict(size=15, family='Arial', color='black'), tickfont=dict(size=10, family='Arial', color='black')),
                      xaxis=dict(title_font=dict(size=15,family='Arial', color='black'), tickfont=dict(size=15, family='Arial', color='black'),
                        ticktext=[f'<b>{tick}</b>' for tick in df_temp[labelx]], tickvals=list(range(len(df_temp[labelx]))),),legend=dict(
            font=dict(size=10, family='Arial',color='black')))
    fig.add_annotation(# text="(c)", 
    text="(b)", xref="paper", yref="paper", x=-0.05, y=1.1, showarrow=False, font=dict(size=20, color="black", family="Arial, sans-serif"), align="left", valign="top")
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.write_image(path, scale=4)
    return 

def get_cfm(df_test,y, path):
    
    """
    Generates a confusion matrix heatmap based on the provided DataFrame and saves it to the specified path.
    """
    
    if y=='memory_efficiency_%':
        suffix='memory'
    else:
        suffix='computetime'
    c = confusion_matrix(df_test['actual_bucket_'+suffix], df_test['predicted_bucket_'+suffix], labels=['0-5', '6-10', '11-20', '21-50', '>50'])
    plt.figure(figsize=(14,6))
    cmap=sns.color_palette("light:b", as_cmap=True) 
    ax = sns.heatmap(c.T, annot=True, cmap=cmap, fmt='d', xticklabels=['0-5', '6-10', '11-20', '21-50', '>50'], yticklabels=['0-5', '6-10', '11-20', '21-50', '>50'], cbar=False, annot_kws={"fontsize":15})
    ax.set(xlabel = 'GroundTruth', ylabel = 'Model Predictions')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.annotate('(a)', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=15, ha='left', va='top')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlabel('GroundTruth', fontweight='bold')
    plt.ylabel('Model Predictions', fontweight='bold')
    plt.savefig(path)
    
    r1, r2, r3, r4, r5 = c[0][0]/sum(c[0]), c[1][1]/sum(c[1]), c[2][2]/sum(c[2]), c[3][3]/sum(c[3]), c[4][4]/sum(c[4])
    p1, p2, p3, p4, p5 = c[0][0]/sum(c[:,0]), c[1][1]/sum(c[:,1]), c[2][2]/sum(c[:,2]), c[3][3]/sum(c[:,3]), c[4][4]/sum(c[:,4])
    n1, n2, n3, n4, n5 = sum(c[0]), sum(c[1]), sum(c[2]), sum(c[3]), sum(c[4])
    total = np.sum(c)
    WeightedRecall = (n1/total*r1)+(n2/total*r2)+(n3/total*r3)+(n4/total*r4)+(n5/total*r5)
    WeightedPrec = (n1/total*p1)+(n2/total*p2)+(n3/total*p3)+(n4/total*p4)+(n5/total*p5)

    upjobs = sum(c[1:,0])+sum(c[2:,1])+sum(c[3:,2])+sum(c[4:,3])
    print('recall', r1, r2, r3, r4, r5)
    print('avg recall',(r1+r2+r3+r4+r5)/5)
    print('weighted recall', WeightedRecall)
    
    print('precision', p1, p2, p3, p4, p5)
    print('avg precision', (p1+p2+p3+p4+p5)/5)
    print('weighted precision', WeightedPrec)
    print('under-provisioning rate',upjobs/total)
    
    return

def pre_process(df_modelling, y, ybucket, dump_path, experiment_suffix):
    
    """
    Preprocesses the input DataFrame for modeling by filtering, bucketing, and splitting into training and validation sets. 
    Split is done based sequentially to ensure that the model is trained on past data and validated on future data.
    """
    
    df_modelling['ComputeUtil_levels'] = df_modelling['compute_utilised_time_hrs'].apply(lambda x:'High' if x>0.0167 else 'Low') # 1 minute as threshold
    bec_path = os.path.join(dump_path, 'encoders','Regressor_bec_all'+experiment_suffix+'.joblib')
    ss_path = os.path.join(dump_path, 'encoders', 'Regressor_ss_all'+experiment_suffix+'.joblib')
    os.makedirs(os.path.dirname(bec_path), exist_ok=True)
    val_size = 0.15
    df_modelling = df_modelling[df_modelling['ComputeUtil_levels']=='High'] 
    df_modelling[ybucket] = df_modelling[y].apply(bucketing)
    df_train, df_val = pd.DataFrame(), pd.DataFrame()
    users = df_modelling['id_user'].unique()
    for userid in users:
        df_user = df_modelling[df_modelling['id_user']==userid]
        df_user.sort_values(by=['time_submit'], inplace=True)
        total = df_user.shape[0]
        train_idx = int((1-val_size)*total)
        df_train = pd.concat([df_train, df_user.iloc[:train_idx,:]], ignore_index=True)
        df_val = pd.concat([df_val, df_user.iloc[train_idx:, :]], ignore_index=True)
    df_train = df_train[(df_train[y]>0.1) & (df_train[y]<=100)]
    print('Train: {} and Validation: {}'.format(df_train.shape, df_val.shape))
    y_train, y_val = df_train[y].values/100, df_val[y].values/100
    return df_train, df_val, y_train, y_val

def train(df_data, dump_path, experiment_suffix, y, data_balance):
    """
    Trains a classification model. Then trains regressions models for etiher memory efficiency or compute time utilization based on the target variable specified by `y`.
    """
    
    user_freq = dict(df_data.groupby('id_user')['id_job'].count())
    df_data['user_count'] = df_data['id_user'].apply(lambda x:user_freq[x])
    df_data = df_data[df_data['user_count']>199]
    starttime = time.time()
    _ = classification_model(df_data, dump_path)
    endtime = time.time()
    print(f"Training time classification model: {endtime - starttime} seconds")
    print(df_data.columns)
    print(df_data.shape)
    df_data = df_data[['id_job', 'id_user', 'id_assoc', 'id_qos', 'id_group', 'partition', 'account', 'constraints', 'wait_time', 
                                 'time_submit', 'job_name', 'cpus_req', 'gpu_req', 'wall_time_hrs', 'allocated_memory_gb', 'compute_utilised_time_hrs', 
                                 'used_memory_gb', 'txt_based_mem', 'txt_based_time', 'job_array_idx', 'memory_efficiency_%', '%_UtilisedTime', 
                                 'last_memuse_gb', 'last_timeuse_hrs', 'mem_rollingmean','time_rollingmean', 'rollingmean_3mem', 'rollingmean_3time', 
                                 'rollingmean_7mem', 'rollingmean_7time', 'rollingmean_10mem', 'rollingmean_10time', 'last_memuse_gb%', 
                                 'last_timeuse_hrs%', 'mem_rollingmean%', 'time_rollingmean%', 'rollingmean_3mem%', 'rollingmean_3time%', 
                                 'rollingmean_7mem%', 'rollingmean_7time%', 'rollingmean_10mem%', 'rollingmean_10time%']]
    
    if y=='memory_efficiency_%':
        ybucket = 'actual_bucket_memory'
    elif y=='%_UtilisedTime':
        ybucket = 'actual_bucket_computetime'
    
    categ_cols = ['id_user', 'account', 'id_assoc']
    keep_cols = categ_cols + ['wait_time', 'time_submit', 'cpus_req', 'gpu_req', 'wall_time_hrs', 'allocated_memory_gb', 
                              'job_array_idx', 'txt_based_mem', 'txt_based_time', 'last_memuse_gb', 'last_timeuse_hrs',
        'mem_rollingmean', 'time_rollingmean', 'rollingmean_3mem', 'rollingmean_3time', 'rollingmean_7mem', 'rollingmean_7time',
        'rollingmean_10mem', 'rollingmean_10time', 'last_memuse_gb%', 'last_timeuse_hrs%', 'mem_rollingmean%', 'time_rollingmean%',
        'rollingmean_3mem%', 'rollingmean_3time%', 'rollingmean_7mem%', 'rollingmean_7time%', 'rollingmean_10mem%', 'rollingmean_10time%']
    
    if data_balance:
        model_path = os.path.join(dump_path, 'models', 'RandomForest_balanced_all_'+experiment_suffix+'.joblib')
        enc_path = os.path.join(dump_path, 'encoders', 'Regressor_bec_balanced_all_'+experiment_suffix+'.joblib')
    else:
        model_path = os.path.join(dump_path, 'models', 'RandomForest_all'+experiment_suffix+'.joblib')
        enc_path = os.path.join(dump_path, 'encoders', 'Regressor_bec_all'+experiment_suffix+'.joblib')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    df_train, df_val, y_train, y_val = pre_process(df_data, y, ybucket, dump_path, experiment_suffix)
    df_train_all = pd.concat([df_train, df_val], ignore_index=True)
    binary_encoder = bec(cols=categ_cols)
    if data_balance:
        df_train = data_balancing(df_train, y, ybucket, combined=False)
        x_train = binary_encoder.fit_transform(df_train[keep_cols]).values
        x_val = binary_encoder.transform(df_val[keep_cols]).values
        y_train = df_train[y].values/100
        y_val = df_val[y].values/100
        starttime = time.time()
        mae, rmse, mape, y_pred, rf_reg, rfparams = RandomForest(x_train, y_train, x_val, y_val)
        print('Random Forest is ', mape, mae, rmse)
        endtime = time.time()
        print(f"Training time on trainval data: {endtime - starttime} seconds")
        # mae, rmse, mape, y_pred, xgb_reg, xgbparams = XgBoostReg(x_train, y_train, x_val, y_val)
        # print('XgBoost is ', mape, mae, rmse)
        # mae, rmse, mape, y_pred, lgbm_reg, lgbmparams = LightGBM(x_train, y_train, x_val, y_val)
        # print('LGBM is ', mape, mae, rmse)
        # mae, rmse, mape, y_pred, lr_reg = LinearReg(x_train, y_train, x_val, y_val)
        # print('LR is ', mape, mae, rmse)
        # mae, rmse, mape, y_pred, lasso_reg, lassoparams = LassoReg(x_train, y_train, x_val, y_val)
        # print('Lasso is ', mape, mae, rmse)
        
        #training on the all data
        starttime = time.time()
        rf_reg = RandomForestRegressor(n_jobs = 18, random_state=42)
        rf_reg.set_params(**rfparams)
        df_train_all = data_balancing(df_train_all, y, ybucket, combined=True)
        x_train_all = binary_encoder.fit_transform(df_train_all[keep_cols]).values
        y_train_all = df_train_all[y].values/100
        rf_reg.fit(x_train_all, y_train_all)
        endtime = time.time()
        print(f"Training time on all data: {endtime - starttime} seconds")
        
    else:
        starttime = time.time()
        x_train = binary_encoder.fit_transform(df_train[keep_cols]).values
        x_val = binary_encoder.transform(df_val[keep_cols]).values
        mae, rmse, mape, y_pred, rf_reg, rfparams = RandomForest(x_train, y_train, x_val, y_val)
        endtime = time.time()
        print(f"Training time on trainval data: {endtime - starttime} seconds")
        print('Random Forest is ', mape, mae, rmse)
        # mae, rmse, mape, y_pred, xgb_reg, xgbparams = XgBoostReg(x_train, y_train, x_val, y_val)
        # print('XgBoost is ', mape, mae, rmse)
        # mae, rmse, mape, y_pred, lgbm_reg, lgbmparams = LightGBM(x_train, y_train, x_val, y_val)
        # print('LGBM is ', mape, mae, rmse)
        # mae, rmse, mape, y_pred, lr_reg= LinearReg(x_train, y_train, x_val, y_val)
        # print('LR is ', mape, mae, rmse)
        # mae, rmse, mape, y_pred, lasso_reg, lassoparams = LassoReg(x_train, y_train, x_val, y_val)
        # print('Lasso is ', mape, mae, rmse)
        x_train_all = binary_encoder.fit_transform(df_train_all[keep_cols]).values
        y_train_all = df_train_all[y].values/100
        starttime = time.time()
        rf_reg = RandomForestRegressor(n_jobs = 18, random_state=42)
        rf_reg.set_params(**rfparams)
        rf_reg.fit(x_train_all, y_train_all)
        endtime = time.time()
        print(f"Training time on all data: {endtime - starttime} seconds")
    
    joblib.dump(rf_reg, model_path, compress = ('lzma', 3))
    joblib.dump(binary_encoder, enc_path)    
    y_pred = rf_reg.predict(x_val)
    y_pred = y_pred*100
    y_val = y_val*100
    
    mapping1 = {'0-5':0.05, '6-10':0.10, '11-20':0.20, '21-50':0.50, '>50':1}
    mapping2 = {'0-5':0.1, '6-10':0.20, '11-20':0.50, '21-50':1, '>50':1}
    
    df_test = df_val
    if y=='memory_efficiency_%':
        df_test['predicted_memory_efficiency'] = y_pred
        df_test['predicted_bucket_memory'] = list(map(bucketing, y_pred))
        df_test['predicted_memory_gb'] = df_test.apply(lambda x:mapping1[x['predicted_bucket_memory']]*x['allocated_memory_gb'], axis=1)
        df_test['savings_memory_gb'] = df_test.apply(lambda x:round(x['allocated_memory_gb']-x['predicted_memory_gb'],2) if x['predicted_memory_gb']>x['used_memory_gb'] else 0, axis=1)
        df_test['predicted_memory_gb_1bucket'] = df_test.apply(lambda x:mapping2[x['predicted_bucket_memory']]*x['allocated_memory_gb'], axis=1)
        df_test['savings_memory_gb_1bucket'] = df_test.apply(lambda x:round(x['allocated_memory_gb']-x['predicted_memory_gb_1bucket'],2) if x['predicted_memory_gb_1bucket']>x['used_memory_gb'] else 0, axis=1)
    else:
        df_test['predicted_computetime'] = y_pred
        df_test['predicted_bucket_computetime'] = list(map(bucketing, y_pred))
        df_test['predicted_computetime_hrs'] = df_test.apply(lambda x:mapping1[x['predicted_bucket_computetime']]*x['wall_time_hrs'], axis=1)
        df_test['improved_computetime_hrs'] = df_test.apply(lambda x:round(x['wall_time_hrs']-x['predicted_computetime_hrs'],2) if x['predicted_computetime_hrs']>x['compute_utilised_time_hrs'] else 0, axis=1)
        df_test['predicted_computetime_hrs_1bucket'] = df_test.apply(lambda x:mapping2[x['predicted_bucket_computetime']]*x['wall_time_hrs'], axis=1)
        df_test['improved_computetime_hrs_1bucket'] = df_test.apply(lambda x:round(x['wall_time_hrs']-x['predicted_computetime_hrs_1bucket'],2) if x['predicted_computetime_hrs_1bucket']>x['compute_utilised_time_hrs'] else 0, axis=1)
    
    get_pr(df_test, y, path = dump_path+'/prf_'+ experiment_suffix+'.png')
    print('path to prf plot', dump_path+'/prf_'+ experiment_suffix+'.png')
    get_cfm(df_test,y, path = dump_path+'/cfm_'+experiment_suffix+'.png')
    print('path to cfm plot', dump_path+'/cfm_'+experiment_suffix+'.png')
    
    if y=='memory_efficiency_%':
        max_ypredict = max(df_test[df_test[y]<1]['predicted_memory_efficiency'])
        df_test = df_test[df_test['predicted_memory_efficiency']!=max_ypredict]
        max_ypredict = max(df_test[df_test[y]<1]['predicted_memory_efficiency'])
        df_test = df_test[df_test['predicted_memory_efficiency']!=max_ypredict]
    
    if ybucket=='actual_bucket_computetime':
        merge_total_percentage(df_test, 'account', 'Compute Time (hrs) improvement by account', 'improved_computetime_hrs', dump_path+'/TIMEsavings_'+experiment_suffix+'.png')
        merge_total_percentage(df_test, 'account', 'Compute Time (hrs) improvement by account', 'improved_computetime_hrs_1bucket', dump_path+'/TIMEsavings1bktup_'+experiment_suffix+'.png')
        print('path to the time savings plot', dump_path+'/TIMEsavings_'+experiment_suffix+'.png')
        print('path to 1bucket time savings plot', dump_path+'/TIMEsavings1bktup_'+experiment_suffix+'.png')
    else:
        merge_total_percentage(df_test, 'account', 'Memory (gb) savings by account', 'savings_memory_gb', dump_path+'/MEMsavings_'+experiment_suffix+'.png')
        merge_total_percentage(df_test, 'account', 'Memory (gb) savings by account', 'savings_memory_gb_1bucket', dump_path+'/MEMsavings1bktup_'+experiment_suffix+'.png')
        print('path to the savings plot', dump_path+'/MEMsavings_'+experiment_suffix+'.png')
        print('path to 1bucket savings plot', dump_path+'/MEMsavings1bktup_'+experiment_suffix+'.png')

    
    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].axis('off')
    axs[1].axis('off')
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(dump_path + '/combined_savings'+experiment_suffix+'.png', dpi=500)
    return


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='./train.csv')
    args.add_argument('--data_balance', type=bool, default=False, help='Whether to balance the data or not')
    args.add_argument('--y', type=str, default='%_UtilisedTime', help='Target variable for regression (eg. %_UtilisedTime, memory_efficiency_%)')
    args.add_argument('--dump_path', type=str, default='./results')
    args = args.parse_args()
    starttime = time.time()
    df_modelling = pd.read_csv(args.data_path)
    cols_remove = [col for col in df_modelling.columns if 'unnamed' in col.lower()]
    df_modelling.drop(columns=cols_remove, inplace=True)
    data_balance, y, dump_path = args.data_balance, args.y, args.dump_path
    
    if data_balance and y=='memory_efficiency_%':
        experiment_suffix = 'MEMwithsampling'
    elif not data_balance and y =='memory_efficiency_%':
        experiment_suffix = 'MEMwithoutsampling'
        
    elif data_balance and y=='%_UtilisedTime':
        experiment_suffix = 'ComputeTimewithsampling'
    elif not data_balance and y=='%_UtilisedTime':
        experiment_suffix= 'ComputeTimewithoutsampling'
        
    dump_path = os.path.join(dump_path, experiment_suffix)
    os.makedirs(dump_path, exist_ok=True)
    train(df_modelling, dump_path, experiment_suffix, y, data_balance)
    print('Total time taken for the training with config {} {} is {} seconds'.format(y, data_balance, round(time.time()-starttime,2)))
