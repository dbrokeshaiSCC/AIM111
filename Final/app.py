import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_core import load_data,evaluate,filter_data,FEATURES,ORDER,PALETTE
st.set_page_config(page_title='Breast Cancer Detection',layout='wide')
st.title('Breast Cancer Detection and Data Analysis Using Machine Learning')
st.caption('AIM111 | Final Project | Educational use only')
st.warning('Not a clinical diagnostic tool. No patient uploads or medical recommendations.')
@st.cache_data
def dataset(): return load_data()
@st.cache_data
def results(df): return evaluate(df)
try: df=dataset()
except Exception as exc:
    st.error(str(exc)); st.stop()
eda=filter_data(df,'All')
def show(fig):
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
summary,exploration,evaluation=st.tabs(['Dataset Overview','Feature Exploration','SVM Model Evaluation'])
with summary:
    st.header('Dataset Overview')
    for col,label,val in zip(st.columns(4),['Observations','Features','Missing values','Duplicates'],[len(df),len(FEATURES),int(df.isna().sum().sum()),int(df.duplicated().sum())]): col.metric(label,val)
    st.caption('Validated against the Wisconsin Diagnostic reference dataset. No cleaning or imputation required. Target: 0 malignant, 1 benign.')
    fig,ax=plt.subplots(figsize=(6,3)); counts=eda.Diagnosis.value_counts().reindex(ORDER); ax.bar(counts.index,counts.values,color=[PALETTE[x] for x in ORDER]); ax.bar_label(ax.containers[0]); ax.set(title='Class distribution',ylabel='Observations'); show(fig)
    st.caption('Class counts describe the historical dataset, not population prevalence.')
    st.dataframe(df.head(10)); st.subheader('Schema and feature ranges'); st.dataframe(pd.DataFrame({'Type':df.dtypes.astype(str),'Missing':df.isna().sum(),'Min':df.min(),'Max':df.max()})); st.subheader('Descriptive statistics'); st.dataframe(df[FEATURES].describe().T)
with exploration:
    st.header('Feature Exploration')
    c1,c2,c3=st.columns(3)
    x=c1.selectbox('X-axis feature',FEATURES,index=FEATURES.index('mean radius'))
    y=c2.selectbox('Y-axis feature',FEATURES,index=FEATURES.index('worst concave points'))
    diagnosis=c3.selectbox('Diagnosis filter',['All']+ORDER)
    subset=filter_data(df,diagnosis); st.write(f'Selected observations: {len(subset)}')
    fig,ax=plt.subplots(figsize=(8,4)); sns.scatterplot(data=subset,x=x,y=y,hue='Diagnosis',palette=PALETTE,ax=ax); ax.set_title(f'{x} vs {y}'); show(fig)
    st.caption('Each point is one observation; the diagnosis filter changes the displayed data only, not model training.')
    fig,ax=plt.subplots(figsize=(7,3)); sns.boxplot(data=subset,x='Diagnosis',y=y,ax=ax); ax.set_title(f'{y} by diagnosis'); show(fig)
    st.caption('Box plots summarize the median, spread and potential outliers.')
    st.dataframe(subset.groupby('Diagnosis')[list(dict.fromkeys([x,y]))].agg(['mean','median','std']))
    fig,ax=plt.subplots(figsize=(13,11)); sns.heatmap(subset[FEATURES].corr(),vmin=-1,vmax=1,center=0,cmap='vlag',ax=ax,cbar_kws={'label':'Pearson r'}); ax.set_title('Feature correlation heatmap'); show(fig)
    st.caption('Pearson correlations reflect the selected subset; correlation does not establish causation.')
with evaluation:
    st.header('SVM Model Evaluation')
    metrics,cm,report,ntrain,ntest,pred=results(df)
    st.write(f'Linear SVC + StandardScaler | Stratified 70/30 split | Seed 42 | Training: {ntrain} | Testing: {ntest}')
    for col,(label,value) in zip(st.columns(4),metrics.items()): col.metric(label,f'{value:.2%}')
    fig,ax=plt.subplots(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=ORDER,yticklabels=ORDER,cbar=False,ax=ax); ax.set(xlabel='Predicted diagnosis',ylabel='Actual diagnosis',title='Held-out confusion matrix'); show(fig)
    tp,fn,fp,tn=cm.ravel(); st.caption(f'Malignant is the positive class (0). TP={tp}, FN={fn}, FP={fp}, TN={tn}. False negatives are malignant observations predicted benign; false positives are benign observations predicted malignant.')
    fig,ax=plt.subplots(figsize=(7,3)); ax.barh(list(metrics),list(metrics.values()),color='#297b91'); ax.set(xlim=(0,1.1),xlabel='Score',title='Held-out performance'); ax.bar_label(ax.containers[0],fmt='%.3f'); show(fig)
    st.caption('Precision, recall and F1 use malignant as the positive class. Accuracy measures all correct classifications.')
    st.dataframe(report.round(4))
    st.info('Limitations: small historical dataset, single held-out split, no external or prospective validation. False negatives and false positives may have serious consequences in clinical settings. These results do not establish clinical reliability.')
st.markdown('Sources: [UCI dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) · [scikit-learn dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)')
