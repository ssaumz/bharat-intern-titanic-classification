# Titanic Classification Model

## Project Overview

This project aims to build a predictive model to determine the likelihood of survival for passengers on the Titanic using data science techniques in Python. The project involves data preprocessing, exploratory data analysis (EDA), and training a machine learning model for prediction.

## Table of Contents

1. [Project Description](#project-description)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Code Explanation](#code-explanation)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Model Training](#model-training)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Project Description

The project involves the following key steps:

1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and preparing the data for analysis.
2. **Exploratory Data Analysis (EDA)**: Visualizing the data to understand patterns and relationships, including the creation of bar graphs for survival rates and passenger sex.
3. **Model Training**: Building and training a predictive model to determine the likelihood of survival using machine learning algorithms.

### Features

- Data cleaning and preprocessing
- Visualization of survival rates by different categories
- Training and evaluating a predictive model

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Data visualization
- **Scikit-learn**: Machine learning library

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or any Python IDE

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Install Dependencies**

   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

3. **Download the Dataset**

   Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place it in the project directory.

## Usage

1. **Open the Jupyter Notebook**

   ```bash
   jupyter notebook titanic_survival_prediction.ipynb
   ```

2. **Run the Notebook**

   Execute the cells in the notebook to preprocess the data, perform EDA, and train the model.

## Project Structure

```
titanic-survival-prediction/
├── data/
│   └── train.csv
├── titanic.ipynb
├── README.md
```

## Code Explanation

### Data Preprocessing

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('data/train.csv')

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'])

# Drop irrelevant columns
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
```

### Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Bar graph for survival rates
sns.countplot(x='Survived', data=data)
plt.title('Survival Counts')
plt.show()

# Bar graph for survival rates by sex
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survival Counts by Sex')
plt.show()
```

### Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

-**Saumya Poojari** - [saumya.poojarii7@gmail.com]

-LinkedIn - https://www.linkedin.com/in/ssaumz/

Feel free to reach out with any questions or feedback!

---

Thank you for your interest in my Titanic Survival Prediction project. I hope you find it informative and useful!


