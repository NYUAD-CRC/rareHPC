import argparse
import os 
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def classification_model(df_modelling, dump_path):
    """
    Function to load the classification model and predict the compute utilization category (High/Low) for the given dataframe.
    It also calculates and prints the recall, precision, and accuracy of the model on the test data, and saves the confusion matrix plot."""
    
    dump_path = os.path.join(dump_path, 'classification_model')
    
    df_model_sel = df_modelling[['id_job', 'id_user', 'id_assoc', 'id_qos', 'id_group', 'constraints', 'partition', 'account', 'job_name', 'wait_time', 
                             'wall_time_hrs', 'allocated_memory_gb', 'time_submit', 'cpus_req', 'gpu_req', 'compute_utilised_time_hrs', 'used_memory_gb', 
                             'last_memuse_gb', 'last_timeuse_hrs', 'rollingmean_3mem', 'rollingmean_3time', 'rollingmean_7mem', 'rollingmean_7time', 
                             'rollingmean_10mem', 'rollingmean_10time','txt_based_mem', 'txt_based_time', 'job_array_idx', 'last_memuse_gb%', 'last_timeuse_hrs%', 
                             'mem_rollingmean%', 'time_rollingmean%', 'rollingmean_3mem%', 'rollingmean_3time%', 'rollingmean_7mem%', 
                             'rollingmean_7time%', 'rollingmean_10mem%', 'rollingmean_10time%']]
    
    df_model_sel['computeTime_bucket'] = df_model_sel['compute_utilised_time_hrs'].apply(lambda x:'High' if x>(1/60) else 'Low')
    bec_path  = os.path.join(dump_path, 'encoders', 'ClassFier_bec_alldata'+'.joblib')
    model_path = os.path.join(dump_path, 'models', 'ClassFier_LGBM_alldata'+'.joblib')
    if not os.path.exists(bec_path) or not os.path.exists(model_path):
        raise Exception('Classification model details not found. Please train the model first using train.py')
    remove_cols = ['job_name', 'compute_utilised_time_hrs', 'used_memory_gb', 'computeTime_bucket', 'constraints', 'partition', 'id_group', 'id_qos','id_job']
    keep_cols = [col for col in df_model_sel.columns if col not in remove_cols]

    y_all = df_model_sel[['compute_utilised_time_hrs','used_memory_gb', 'computeTime_bucket']]
    
    df_test = df_model_sel[keep_cols]
    y_test =y_all['computeTime_bucket']
        
    be=joblib.load(bec_path.replace('_test', ''))
    model=joblib.load(model_path.replace('_test', ''))
    df_res_test = be.transform(X=df_test)
    pred_utilizCategory = model.predict(df_res_test)
    
    cfm = confusion_matrix(y_test.values, pred_utilizCategory, labels=['Low','High'])
    recall = recall_score(y_test.values, pred_utilizCategory, pos_label='High')
    precision = precision_score(y_test.values, pred_utilizCategory, pos_label='High')
    accuracy = accuracy_score(y_test.values, pred_utilizCategory)
    print('Recall for Compute Utilization High/Low:', round(recall*100, 2), '%')
    print('Precision for Compute Utilization High/Low:', round(precision*100, 2), '%')
    print('Accuracy for Compute Utilization High/Low:', round(accuracy*100, 2), '%')
    f1 = f1_score(y_test.values, pred_utilizCategory, pos_label='High')
    cmap=sns.color_palette("light:b", as_cmap=True) 
    sns.heatmap(cfm.T, annot=True, cmap=cmap, fmt='d', xticklabels=['Low','High'], yticklabels=['Low','High'], cbar=False,annot_kws={"fontsize":15})
    plt.xlabel('GroundTruth', fontweight='bold')
    plt.ylabel('Model Predictions', fontweight='bold')
    plt.savefig(dump_path+'/Classifier_cfm_test.png')
    df_modelling['pred_utilizCategory'] = pred_utilizCategory
    return df_modelling

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

def get_pr(df, y, path):
    
    """
    Function to calculate precision, recall, and F1 score for each category of the target variable and plot them as a grouped bar chart.
    """
    
    if y=='memory_efficiency_%':
        suffix='memory'
    else:
        suffix='computetime'
    
    categories = ['0-5', '6-10', '11-20', '21-50', '>50']
    prec = precision_score(df['actual_bucket_'+suffix], df['predicted_bucket_'+suffix], labels = categories, average=None)
    rec = recall_score(df['actual_bucket_'+suffix], df['predicted_bucket_'+suffix], labels = categories, average=None)
    fscore = f1_score(df['actual_bucket_'+suffix], df['predicted_bucket_'+suffix], labels = categories, average=None)
    rec = [round(item*100,2) for item in rec]
    prec = [round(item*100,2) for item in prec]
    fscore = [round(item*100,2) for item in fscore]
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
    Function to calculate and plot the confusion matrix for the classification results.
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
    # ax.annotate('(b)', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=15, ha='left', va='top')
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
    print('weighted recall', round(WeightedRecall*100, 2), '%')
    print('weighted precision', round(WeightedPrec*100, 2), '%')
    print('under-provisioning rate', round(upjobs/total*100, 2), '%')
    return

def plot_clean2(df_plot, labelx, title, labely, save_path, topk1=15, topk2=10):
    """
    Function to plot a grouped bar chart showing total savings and percentage savings for a given label.
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
    Function to merge total savings and percentage savings for a given label and plot them as a grouped bar chart.
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
    plot_clean2(merged_df, labelx, title, labely, save_path)

def test(df_data, dump_path, y, experiment_suffix):
    """
    Function to test the classification model, regression model for the specified target variable and evaluate their performance.
    """
    
    user_freq = dict(df_data.groupby('id_user')['id_job'].count())
    df_data['user_count'] = df_data['id_user'].apply(lambda x:user_freq[x])
    df_data = df_data[df_data['user_count']>199]
    df_data = classification_model(df_data, dump_path)
    df_data = df_data[df_data['pred_utilizCategory']=='High']
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
    dump_path = os.path.join(dump_path, experiment_suffix)
    categ_cols = ['id_user', 'account', 'id_assoc']
    keep_cols = categ_cols + ['wait_time', 'time_submit', 'cpus_req', 'gpu_req', 'wall_time_hrs', 'allocated_memory_gb', 
                              'job_array_idx', 'txt_based_mem', 'txt_based_time', 'last_memuse_gb', 'last_timeuse_hrs',
        'mem_rollingmean', 'time_rollingmean', 'rollingmean_3mem', 'rollingmean_3time', 'rollingmean_7mem', 'rollingmean_7time',
        'rollingmean_10mem', 'rollingmean_10time', 'last_memuse_gb%', 'last_timeuse_hrs%', 'mem_rollingmean%', 'time_rollingmean%',
        'rollingmean_3mem%', 'rollingmean_3time%', 'rollingmean_7mem%', 'rollingmean_7time%', 'rollingmean_10mem%', 'rollingmean_10time%']
    model_path = os.path.join(dump_path, 'models', 'RandomForest_balanced_all_'+experiment_suffix+'.joblib')
    enc_path = os.path.join(dump_path, 'encoders', 'Regressor_bec_balanced_all_'+experiment_suffix+'.joblib')
    rf_reg = joblib.load(model_path.replace('_test',''))
    bec= joblib.load(enc_path.replace('_test',''))
    
    x_test = bec.transform(df_data[keep_cols]).values
    
    y_pred = rf_reg.predict(x_test)
    y_pred = y_pred*100
    
    mapping1 = {'0-5':0.05, '6-10':0.10, '11-20':0.20, '21-50':0.50, '>50':1}
    mapping2 = {'0-5':0.1, '6-10':0.20, '11-20':0.50, '21-50':1, '>50':1}
    df_data[ybucket] = df_data[y].apply(bucketing)
    
    if y=='memory_efficiency_%':
        df_data['predicted_memory_efficiency'] = y_pred
        df_data['predicted_bucket_memory'] = list(map(bucketing, y_pred))
        df_data['predicted_memory_gb'] = df_data.apply(lambda x:mapping1[x['predicted_bucket_memory']]*x['allocated_memory_gb'], axis=1)
        df_data['savings_memory_gb'] = df_data.apply(lambda x:round(x['allocated_memory_gb']-x['predicted_memory_gb'],2) if x['predicted_memory_gb']>x['used_memory_gb'] else 0, axis=1)
        df_data['predicted_memory_gb_1bucket'] = df_data.apply(lambda x:mapping2[x['predicted_bucket_memory']]*x['allocated_memory_gb'], axis=1)
        df_data['savings_memory_gb_1bucket'] = df_data.apply(lambda x:round(x['allocated_memory_gb']-x['predicted_memory_gb_1bucket'],2) if x['predicted_memory_gb_1bucket']>x['used_memory_gb'] else 0, axis=1)
    else:
        df_data['predicted_computetime'] = y_pred
        df_data['predicted_bucket_computetime'] = list(map(bucketing, y_pred))
        df_data['predicted_computetime_hrs'] = df_data.apply(lambda x:mapping1[x['predicted_bucket_computetime']]*x['wall_time_hrs'], axis=1)
        df_data['improved_computetime_hrs'] = df_data.apply(lambda x:round(x['wall_time_hrs']-x['predicted_computetime_hrs'],2) if x['predicted_computetime_hrs']>x['compute_utilised_time_hrs'] else 0, axis=1)
        df_data['predicted_computetime_hrs_1bucket'] = df_data.apply(lambda x:mapping2[x['predicted_bucket_computetime']]*x['wall_time_hrs'], axis=1)
        df_data['improved_computetime_hrs_1bucket'] = df_data.apply(lambda x:round(x['wall_time_hrs']-x['predicted_computetime_hrs_1bucket'],2) if x['predicted_computetime_hrs_1bucket']>x['compute_utilised_time_hrs'] else 0, axis=1)
    
    get_pr(df_data, y, path = dump_path+'/prf_'+ experiment_suffix+'_test.png')
    print('path to prf plot', dump_path+'/prf_'+ experiment_suffix+'_test.png')
    get_cfm(df_data,y, path = dump_path+'/cfm_'+experiment_suffix+'_test.png')
    print('path to cfm plot', dump_path+'/cfm_'+experiment_suffix+'_test.png')
    
    
    if ybucket=='actual_bucket_computetime':
        merge_total_percentage(df_data, 'account', 'Compute Time (hrs) improvement by account', 'improved_computetime_hrs', dump_path+'/TIMEsavings_'+experiment_suffix+'_test.png')
        merge_total_percentage(df_data, 'account', 'Compute Time (hrs) improvement by account', 'improved_computetime_hrs_1bucket', dump_path+'/TIMEsavings1bktup_'+experiment_suffix+'_test.png')
        img1 = plt.imread(dump_path+'/TIMEsavings_'+experiment_suffix+'_test.png')
        img2 = plt.imread(dump_path+'/TIMEsavings1bktup_'+experiment_suffix+'_test.png')
    else:
        merge_total_percentage(df_data, 'account', 'Memory (gb) savings by account', 'savings_memory_gb', dump_path+'/MEMsavings_'+experiment_suffix+'_test.png')
        merge_total_percentage(df_data, 'account', 'Memory (gb) savings by account', 'savings_memory_gb_1bucket', dump_path+'/MEMsavings1bktup_'+experiment_suffix+'_test.png')
        print('path to the savings plot', dump_path+'/MEMsavings_'+experiment_suffix+'_test.png')
        print('path to 1bucket savings plot', dump_path+'/MEMsavings1bktup_'+experiment_suffix+'_test.png')
        img1 = plt.imread(dump_path+'/MEMsavings_'+experiment_suffix+'_test.png')
        img2 = plt.imread(dump_path+'/MEMsavings1bktup_'+experiment_suffix+'_test.png')
        
    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Display the images in the subplots without axes
    axs[0].imshow(img1)
    axs[0].axis('off')
    axs[1].imshow(img2)
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(dump_path + '/combined_savings'+experiment_suffix+'_test.png', dpi=500)
    return


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='./test.csv')
    args.add_argument('--y', type=str, help='Target variable for regression (eg. %_UtilisedTime, memory_efficiency_%)', required=True)
    args.add_argument('--dump_path', type=str, default='./')
    args = args.parse_args()
    df_data = pd.read_csv(args.data_path)
    if args.y=='memory_efficiency_%':
        experiment_suffix = 'MEMwithsampling'
    elif args.y=='%_UtilisedTime':
        experiment_suffix = 'ComputeTimewithsampling'
    
    test(df_data, args.dump_path, args.y, experiment_suffix)
