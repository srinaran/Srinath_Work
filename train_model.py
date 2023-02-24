import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


def get_claims_xgb_model(X_train, y_train):
    param_grid = {'n_estimators': [100, 200, 300], 
              'learning_rate': [0.05, 0.1, 0.15],
              'max_depth': [2, 3, 4],
              'min_samples_split': [0.05, 0.1, 0.15],
              'min_samples_leaf': [0.02, 0.05, 0.1]
              }
    
    gbm = XGBClassifier(random_state=314)

    xgb_claims_model = RandomizedSearchCV(gbm, param_grid, cv=3, n_iter=10, n_jobs=-1, random_state=314)

    xgb_claims_model.fit(X_train, y_train)

    return xgb_claims_model


def get_endorsement_xgb_model(X_train, y_train):
    param_grid = {'n_estimators': [100, 200, 300], 
              'learning_rate': [0.05, 0.1, 0.15],
              'max_depth': [2, 3, 4],
              'min_samples_split': [0.05, 0.1, 0.15],
              'min_samples_leaf': [0.02, 0.05, 0.1]
              }
    
    gbm = XGBClassifier(random_state=314)

    xgb_endorse_model = RandomizedSearchCV(gbm, param_grid, cv=3, n_iter=10, n_jobs=-1, random_state=314)

    xgb_endorse_model.fit(X_train, y_train)

    return xgb_endorse_model

