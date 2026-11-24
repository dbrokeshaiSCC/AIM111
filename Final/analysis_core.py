from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
LABELS={0:'Malignant',1:'Benign'}
ORDER=list(LABELS.values())
PALETTE={'Malignant':'#b33c45','Benign':'#297b91'}
FEATURES=list(load_breast_cancer().feature_names)
def load_data(path=None):
    df=pd.read_csv(path or Path(__file__).with_name('breast_cancer_dataset.csv'))
    ref=load_breast_cancer()
    if set(df.columns)!=set(FEATURES+['target']) or df.shape!=(569,31): raise ValueError('Unexpected dataset schema')
    if df.isna().any().any() or df.duplicated().any() or not np.isfinite(df.to_numpy()).all(): raise ValueError('Data quality failure')
    if not np.allclose(df[FEATURES],ref.data) or not np.array_equal(df.target,ref.target): raise ValueError('Dataset differs from reference')
    return df

def evaluate(df):
    a,b,c,d=train_test_split(df[FEATURES],df.target,test_size=.30,random_state=42,stratify=df.target)
    model=Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='linear'))]).fit(a,c)
    score=model.score(b,d)
    pred=model.predict(b)
    metrics={'Accuracy':accuracy_score(d,pred),'Precision (malignant)':precision_score(d,pred,pos_label=0),'Recall (malignant)':recall_score(d,pred,pos_label=0),'F1 (malignant)':f1_score(d,pred,pos_label=0)}
    assert np.isclose(score,metrics['Accuracy'])
    assert np.allclose(model.named_steps['scaler'].mean_,a.mean())
    assert set(a.index).isdisjoint(b.index)
    return metrics,confusion_matrix(d,pred,labels=[0,1]),pd.DataFrame(classification_report(d,pred,labels=[0,1],target_names=ORDER,output_dict=True)).T,len(a),len(b),pred

def filter_data(df,diagnosis):
    eda=df.assign(Diagnosis=df.target.map(LABELS))
    return eda if diagnosis=='All' else eda[eda.Diagnosis==diagnosis]
