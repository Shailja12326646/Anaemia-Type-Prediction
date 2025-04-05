import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv('C:\\Users\\hp\\Desktop\\data science\\diagnosed_cbc_data_v4 (1).csv')
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target (strings)

# Encode target variable (since XGBoost needs numeric labels)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define and train StackingClassifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=10)),
    ('gbdt', GradientBoostingClassifier())
]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=10)
clf.fit(x_train, y_train)

# Train XGBoost modelpython 
xgb_model = xgb.XGBClassifier()
xgb_model.fit(x_train, y_train)

# Save models and label encoder
joblib.dump(clf, 'C:\\Users\\hp\\Desktop\\data science\\stacking_classifier_model.pkl')
joblib.dump(xgb_model, 'C:\\Users\hp\\Desktop\\data science\\xgboost_model.pkl')
joblib.dump(le, 'C:\\Users\\hp\\Desktop\\data science\\label_encoder.pkl')

print("Models and label encoder saved successfully!")