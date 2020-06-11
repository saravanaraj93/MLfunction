# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def apply_model(df):
    array = df.values
    X = array[:, 0:8]
    
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled


