import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class Classifier:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=34)

    def scale(self, scaler_type: int = 0):
        if scaler_type == 0:
            scaler = StandardScaler()
        elif scaler_type == 1:
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        return print("Train/test data scaling is completed.")

    def classify_with_LogReg(self):
        model = LogisticRegression(solver='saga', max_iter=1000)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), \
            recall_score(self.y_test, y_pred), \
            precision_score(self.y_test, y_pred), \
            classification_report(self.y_test, y_pred)

    def classify_with_RF(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), \
            recall_score(self.y_test, y_pred), \
            precision_score(self.y_test,y_pred), \
            classification_report(self.y_test, y_pred)