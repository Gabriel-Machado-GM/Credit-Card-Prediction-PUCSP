# Credit Card Prediction PUCSP 2024.2 Project

## Project Objective
The objective of this project is to predict the likelihood of default payments by credit card clients using a dataset from Kaggle. This dataset contains detailed information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

## Dataset
This dataset is sourced from the Kaggle link: [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).

### Dataset Information
The dataset contains 25 variables, including:
- **ID**: ID of each client
- **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0 to PAY_6**: Repayment status from April 2005 to September 2005
- **BILL_AMT1 to BILL_AMT6**: Amount of bill statement from April 2005 to September 2005
- **PAY_AMT1 to PAY_AMT6**: Amount of previous payment from April 2005 to September 2005
- **default.payment.next.month**: Default payment (1=yes, 0=no)

Inspiration for exploration includes analyzing how the probability of default payment varies by different demographic variables and identifying the strongest predictors of default payment.

## Libraries Used
Below is a summary of the libraries used in this project, including their pros, cons, and ideal use cases with code snippets.

### 1. NumPy
- **Summary**: NumPy is a library for numerical computations in Python.
- **Pros**:
  - Efficient storage and manipulation of large data arrays.
  - Supports a wide range of mathematical operations.
- **Cons**:
  - Requires understanding of array-based programming.
- **Ideal Use**: Handling numerical data and performing mathematical operations.
#### Example Code:
```python
import numpy as np

# Creating an array
data = np.array([1, 2, 3, 4, 5])
```

### 2. Pandas
- **Summary**: Pandas is a data manipulation and analysis library.
- **Pros**:
  - Easy handling of data frames.
  - Powerful data manipulation and aggregation functions.
- **Cons**:
  - Can be memory-intensive with large datasets.
- **Ideal Use**: Data cleaning, manipulation, and analysis.
#### Example Code:
```python
import pandas as pd

# Reading a CSV file
df = pd.read_csv('credit_card_data.csv')

# Displaying the first few rows
print(df.head())
```

### 3. Scikit-learn
- **Summary**: Scikit-learn is a machine learning library.
- **Pros**:
  - Extensive collection of machine learning algorithms.
  - Easy integration with NumPy and Pandas.
- **Cons**:
  - Limited support for deep learning.
- **Ideal Use**: Implementing machine learning models for classification, regression, clustering, etc.
#### Example Code:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 4. Matplotlib
- **Summary**: Matplotlib is a library for creating static, animated, and interactive visualizations.
- **Pros**:
  - Highly customizable plots.
  - Wide range of plot types.
- **Cons**:
  - Can be complex for advanced visualizations.
- **Ideal Use**: Creating visual representations of data.
#### Example Code:
```python
import matplotlib.pyplot as plt

# Creating a simple plot
plt.plot(data)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sample Plot')
plt.show()
```

### 5. Seaborn
- **Summary**: Seaborn is a data visualization library based on Matplotlib.
- **Pros**:
  - Simplifies complex visualizations.
  - Integrates well with Pandas.
- **Cons**:
  - Limited customization compared to Matplotlib.
- **Ideal Use**: Creating statistical graphics.
#### Example Code:
```python
import seaborn as sns

# Creating a heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()
```

### Conclusion
This project leverages powerful libraries to clean, analyze, and create predictive models for credit card default prediction. Each library brings its strengths to the project, ensuring efficient and effective analysis and visualization of the dataset.

For more details, refer to the [README.md](https://github.com/Gabriel-Machado-GM/Credit-Card-Prediction-PUCSP-2024.2-/blob/0fae35b42975f182f228aa045ce338b76111bd25/README.md) in the repository.
