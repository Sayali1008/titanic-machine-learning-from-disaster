# Titanic Machine Learning from Disaster

This repository contains experiments comparing Decision Tree and Logistic Regression classifiers on the famous [Titanic dataset](https://www.kaggle.com/competitions/titanic/data). It demonstrates both built-in and custom implementations of these algorithms and highlights their performance on the dataset.

## Project Structure

```
.
├── decision_tree/
│   └── model.py         # Decision Tree models using Gini impurity and Entropy (scikit-learn)
├── logistic_regression/
│   └── model.py         # Custom logistic regression and scikit-learn logistic regression
├── data_analysis.ipynb  # Exploratory data analysis and preprocessing
└── README.md
```

## Notes

- Decision tree experiments compare Gini impurity vs Entropy splits.
- Logistic regression includes a custom implementation alongside scikit-learn's version.
- The goal is to evaluate and compare model performance on the Titanic survival prediction task.
