from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_using_gini(X_train, y_train):
    cls_gini = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=5, min_samples_leaf=3)
    cls_gini.fit(X_train, y_train)
    return cls_gini

def train_using_entropy(X_train, y_train):
    cls_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=5, min_samples_leaf=3)
    cls_entropy.fit(X_train, y_train)
    return cls_entropy

def predict(X_test, classifier):
    y_pred = classifier.predict(X_test)
    return y_pred

def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()

def preprocess_data(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())    # Fill null Age values with the median
    df = df.dropna(subset=["Embarked"])                 # Drop rows where Embarked has null values

    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1)

    # Manually map categorical features
    sex_mapping = {"male": 0, "female": 1}
    df["Sex"] = df["Sex"].map(sex_mapping)
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    df["Embarked"] = df["Embarked"].map(embarked_mapping)

    df = df.sample(frac = 1)    # Shuffle the data
    return df

def handle_imbalanced_data(df):
    # Undersampling: Removing some majority-class examples to level things out.
    class0 = df[df['Survived'] == 0][:342]
    class1 = df[df['Survived'] == 1]

    # TODO Oversampling: Duplicate minority examples or use something like SMOTE to create synthetic ones.
    
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
    X_test = df_test.copy()

    # Model training - gini impurity
    cls_gini = train_using_gini(X_train, y_train)
    y_val_pred_gini = predict(X_val, cls_gini)
    print(f"Accuracy of Decision tree classifier trained using gini impurity: {accuracy_score(y_val, y_val_pred_gini)*100}")

    # Model training - entropy
    cls_entropy = train_using_entropy(X_train, y_train)
    y_val_pred_entropy = predict(X_val, cls_entropy)
    print(f"Accuracy of Decision tree classifier trained using gini impurity: {accuracy_score(y_val, y_val_pred_entropy)*100}")


    # Plot decision tree
    # plot_decision_tree(classifier, X_train.columns, ['0', '1'])

if __name__ == '__main__':
    main()