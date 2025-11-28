# Titanic Machine Learning from Disaster

> _Goal - Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck._

---

## Understanding the data

### 1. Look at the labels first
- Check how many examples belong to each class. If one class is much smaller than the other, that will affect how we evaluate the model and how we handle sampling later.

### 2. Get a feel for each feature
- For numerical features, look at basic stats: min, max, mean, standard deviation, and a few plots like histograms or box plots. We just want to know the rough shape of the data, see if things look normal or if there are weird outliers.
- For categorical features, look at value counts to see if some categories barely appear.

### 3. Check simple relationships
- Pairwise correlations for numeric features can tell us if two variables carry the same information. 
- Scatterplots or simple crosstabs help a lot. We want to see what features might be useful, noisy, or redundant.

### 4. Look for data quality issues.
- Missing values, inconsistent formats, duplicate rows, typos in categorical labels, extreme outliers, or fields that look constant. With a small dataset, even a few bad points can distort training.

### 5. Inspect how features relate to labels.
- For numeric features, compare value distributions between the two classes. 
- For categorical features, see how class proportions change across categories. This gives us a sense of which features might be informative.

### 6. Think about feature scales and types.
- Some algorithms need us to normalize or standardize numeric variables. 
- Some features might need one-hot encoding. We want a mental map of what preprocessing steps are required.

### 7. Split before we peek too much.
- Before doing anything that might indirectly leak label info (like tuning preprocessing based on the whole dataset), set aside a validation or test set. With small data, model leakage is easy and can mislead us.