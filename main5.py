import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
import joblib

df=pd.read_csv('term_deposit_prediction.csv')

cat_cols=['job','marital','education','default','housing','loan','contact','month','poutcome']
num_cols=['age','balance','day_of_week','duration','campaign','pdays','previous']
for col in cat_cols:
    df[col].fillna('unknown',inplace=True)
for col in num_cols:
    df[col].fillna(df[col].median(),inplace=True)

print(f"Number of missing values in 'term_deposit': {df['term_deposit'].isnull().sum()}")
df['term_deposit'].fillna(df['term_deposit'].mode()[0],inplace=True)

X=df.drop('term_deposit',axis=1)
y=df['term_deposit']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

num_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

cat_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant',fill_value='unknown')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',num_transformer,num_cols),
        ('cat',cat_transformer,cat_cols)
    ])

X_train_pre=preprocessor.fit_transform(X_train)
X_test_pre=preprocessor.transform(X_test)

smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train_pre,y_train)

models={
    'Random Forest':(RandomForestClassifier(),{
        'n_estimators':[100,200],
        'max_depth':[None,10,20],
        'min_samples_split':[2,5],
        'min_samples_leaf':[1,2]
    }),
    'Gradient Boosting':(GradientBoostingClassifier(),{
        'n_estimators':[100,200],
        'learning_rate':[0.01,0.1],
        'max_depth':[3,5]
    }),
    'Logistic Regression':(LogisticRegression(),{
        'C':[0.01,0.1,1,10],
        'penalty':['l1','l2'],
        'solver':['liblinear']
    }),
    'XGBoost':(XGBClassifier(),{
        'n_estimators':[100,200],
        'learning_rate':[0.01,0.1],
        'max_depth':[3,5],
        'subsample':[0.8,1.0],
        'colsample_bytree':[0.8,1.0]
    }),
    'K-Nearest Neighbors':(KNeighborsClassifier(),{
        'n_neighbors':[3,5,7],
        'weights':['uniform','distance'],
        'p':[1,2]
    })
}

best_models={}
for name,(model,param_grid) in models.items():
    grid_search=GridSearchCV(model,param_grid,cv=5,scoring='f1',n_jobs=-1)
    grid_search.fit(X_train_smote,y_train_smote)
    best_model=grid_search.best_estimator_
    y_pred=best_model.predict(X_test_pre)
    y_prob=best_model.predict_proba(X_test_pre)[:,1]
    f1=f1_score(y_test,y_pred)
    roc_auc=roc_auc_score(y_test,y_prob)
    best_models[name]={'model':best_model,'f1':f1,'roc_auc':roc_auc,'best_params':grid_search.best_params_}
    print(f"{name}: F1 Score = {f1:.4f}, ROC AUC = {roc_auc:.4f}, Best Params = {grid_search.best_params_}")

best_model_name=max(best_models,key=lambda x:best_models[x]['f1'])
best_model=best_models[best_model_name]['model']
print(f"Best Model: {best_model_name}, F1 Score: {best_models[best_model_name]['f1']:.4f}, ROC AUC: {best_models[best_model_name]['roc_auc']:.4f}")

joblib.dump(best_model,'best_model.pkl')
joblib.dump(preprocessor,'preprocessor.pkl')

example={
    "age":42,"job":"management","marital":"married","education":"tertiary",
    "default":"no","balance":1500,"housing":"yes","loan":"no",
    "contact":"cellular","day_of_week":3,"month":"may",
    "duration":300,"campaign":2,"pdays":-1,"previous":0,"poutcome":"unknown"
}
new_data=pd.DataFrame([example])
new_data_pre=preprocessor.transform(new_data)
predictions=best_model.predict(new_data_pre)
print(f"Predictions for new data: {predictions}")