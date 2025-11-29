from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class MyLogisticRegression():
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def weight_initialization(self, X):
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0

    # Sigmoid function
    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def compute_gradients(self, X, y, y_pred):
        difference = y_pred - y
        dw = (1 / self.num_samples) * np.dot(X.T, difference)
        db = (1 / self.num_samples) * np.sum(difference)
        return dw, db

    def compute_loss(self, y_true, y_pred):
        # Binary cross entropy loss (Log loss)
        loss = - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return np.mean(loss)

    # Fit the model to training data
    def fit(self, X, y):
        self.num_samples, self.num_features = X.shape
        train_loss_list = []
        train_acc_list = []
        parts = int(self.epochs // 10)

        # Initialize weights and bias
        self.weight_initialization(X)

        # Gradient descent
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(y_pred)

            y_pred_list = np.array([1 if y_pred[i] > 0.5 else 0 for i in range(len(y_pred))])
            acc = accuracy_score(y, y_pred_list)
            train_acc_list.append(acc)

            # Calculate loss
            loss = self.compute_loss(y, y_pred)
            train_loss_list.append(loss)

            # Compute and update model parameters
            dw, db = self.compute_gradients(X, y, y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # if epoch % parts == 0:
            #     print(f"Epoch {epoch}: Train accuracy: {acc} \t Train loss: {loss}")
        
        train_acc = np.mean(train_acc_list)
        train_loss = np.mean(train_loss_list)

        return train_acc, train_loss

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(y_pred)
        return [1 if y_pred[i] > 0.5 else 0 for i in range(len(y_pred))]

def preprocess_data(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())    # Fill null Age values with the median
    df = df.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1)    # Drop unnecessary columns
    # One hot encoding categorical data
    df = pd.get_dummies(df, columns = ['Pclass', 'Sex', 'Embarked'], dtype = int)
    # Shuffle the data
    df = df.sample(frac = 1)
    return df


def handle_imbalanced_data(df):
    # Undersampling: Removing some majority-class examples to level things out.
    class0 = df[df['Survived'] == 0][:342]
    class1 = df[df['Survived'] == 1]

    df_balanced = pd.concat([class0, class1])
    df_balanced = df_balanced.sample(frac = 1)
    return df_balanced

def main():
    df_train = pd.read_csv('../titanic/train.csv')
    df_test = pd.read_csv('../titanic/test.csv')

    # Data preprocessing
    df_train = preprocess_data(df_train)
    df_train = handle_imbalanced_data(df_train)
    df_test = preprocess_data(df_test)

    # Split dataset into train, val and test
    X_df = df_train.iloc[:, df_train.columns != 'Survived']
    y_df = df_train['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], 1))

    # Custom model training
    custom_log_reg_model = MyLogisticRegression()
    custom_log_reg_model.fit(X_train, y_train)
    y_val_pred_custom = custom_log_reg_model.predict(X_val)
    print(f"Accuracy of custom Logistic regression model: {accuracy_score(y_val, y_val_pred_custom)*100}")

    # Model training
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, y_train.ravel())
    y_val_pred = log_reg_model.predict(X_val)
    print(f"Accuracy of Logistic regression model: {accuracy_score(y_val, y_val_pred)*100}")

if __name__ == '__main__':
    main()