# A Spark-Based Big Data Analytics Framework For Competitive Intelligence

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Libraries Used](#libraries-used)
- [Steps Involved](#steps-involved)
  - [1. Data Selection](#1-data-selection)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Data Splitting](#3-data-splitting)
  - [4. Feature Selection](#4-feature-selection)
  - [5. SVM Algorithm](#5-svm-algorithm)
  - [6. Naive Bayes Algorithm](#6-naive-bayes-algorithm)
  - [7. Random Forest Algorithm](#7-random-forest-algorithm)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project presents a Spark-based framework for big data analytics focused on competitive intelligence. It leverages various machine learning algorithms and data processing techniques to analyze and gain insights from large datasets.

## Features
- Data selection and preprocessing
- Feature selection using KMeans clustering
- Machine learning models: Decision Tree, SVM, Naive Bayes, Random Forest
- Performance evaluation and visualization

## Installation
Clone the repository and install the required libraries:

```bash
git clone <repository-url>
cd A-Spark-Based-Big-Data-Analytics-Framework-For-Competitive-Intelligence
pip install -r requirements.txt
```

## Usage
Run the provided script:

```bash
python main.py
```

## Project Structure
```plaintext
├── data.csv
├── main.py
├── requirements.txt
└── README.md
```

## Libraries Used
- TensorFlow
- Neural Structured Learning
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Steps Involved

### 1. Data Selection
Load the dataset and display a sample of the input data.

```python
dataframe = pd.read_csv("data.csv")
print("Samples of our input data")
print(dataframe.head(10))
```

### 2. Preprocessing
Handle missing values and perform label encoding.

```python
# Checking missing values
print("Before Handling Missing Values")
print(dataframe.isnull().sum())

dataframe_2 = dataframe.fillna(0)
print("After Handling Missing Values")
print(dataframe_2.isnull().sum())

# Label encoding
label_encoder = preprocessing.LabelEncoder()
print("Before Label Handling")
print(dataframe_2.head(10))
```

### 3. Data Splitting
Split the data into training and testing sets.

```python
x = dataframe.drop('VersionInformationSize', axis=1)
y = dataframe.VersionInformationSize

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
```

### 4. Feature Selection
Perform feature selection using KMeans clustering and train a Decision Tree classifier.

```python
# KMeans clustering
x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(x[:, 0], x[:, 1], s=50)

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("k-means")
plt.show()

# Decision Tree
dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
dt.fit(x_train, y_train)
dt_prediction = dt.predict(x_test)
print("Decision Tree")
print(metrics.classification_report(y_test, dt_prediction))
print("DT Accuracy is:", accuracy_score(y_test, dt_prediction) * 100, '%')
```

### 5. SVM Algorithm
Scale the data and train an SVM classifier.

```python
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

svclassifier = SVC()
svclassifier.fit(x_train, y_train)
y_pred11 = svclassifier.predict(x_test)
print("SVM Algorithm")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred11))
print("Classification Report:")
print(classification_report(y_test, y_pred11))
print("Accuracy:", accuracy_score(y_test, y_pred11))
```

### 6. Naive Bayes Algorithm
Train a Naive Bayes classifier.

```python
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Naive Bayes Algorithm")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### 7. Random Forest Algorithm
Train a Random Forest classifier.

```python
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
print("Random Forest")
print(metrics.classification_report(y_test, rf_prediction))
print("Random Forest Accuracy is:", accuracy_score(y_test, rf_prediction) * 100, '%')
```

## Results and Visualizations
The project includes various visualizations such as scatter plots for KMeans clustering, confusion matrices, and ROC curves for different algorithms.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.
```
