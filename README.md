# Retail Store Sales: Data Cleaning and Analysis

## Project Overview

This project focuses on cleaning, inspecting, and analyzing a retail store sales dataset containing transaction-level information.

The analysis was performed using Python and covers the complete data-preparation workflow, including:

- Understanding the structure of the dataset
- Identifying missing values
- Checking for duplicate records
- Cleaning incomplete and unreliable data
- Handling missing numerical values
- Detecting potential outliers using IQR and Z-score methods
- Comparing standard scaling and Min-Max scaling
- Examining correlations between numerical variables
- Preparing the dataset for potential machine-learning applications

The purpose of this project is to demonstrate practical data-analysis and preprocessing skills using a real-world-style retail dataset.

## Dataset

The original dataset is named:

```text
retail_store_sales.csv
```

It contains **12,575 rows and 11 columns** before cleaning.

Each row represents a retail transaction.

### Original columns

| Column | Description |
|---|---|
| `Transaction ID` | Unique identifier for each transaction |
| `Customer ID` | Identifier for the customer |
| `Category` | Product category |
| `Item` | Item identifier or item name |
| `Price Per Unit` | Price of one unit of the product |
| `Quantity` | Number of units purchased |
| `Total Spent` | Total amount spent in the transaction |
| `Payment Method` | Method used for payment |
| `Location` | Transaction channel or location |
| `Transaction Date` | Date of the transaction |
| `Discount Applied` | Indicates whether a discount was applied |

## Initial Dataset Summary

Before cleaning, the dataset contained:

- **Rows:** 12,575
- **Columns:** 11
- **Unique customers:** 25
- **Unique categories:** 8
- **Unique items:** 200
- **Payment methods:** 3
- **Locations:** 2
- **Transaction dates:** 1,114
- **Duplicate records:** 0

## Missing Values

The main missing-value issues were found in the following columns:

| Column | Missing values | Approximate percentage |
|---|---:|---:|
| `Item` | 1,213 | 9.65% |
| `Price Per Unit` | 609 | 4.84% |
| `Quantity` | 604 | 4.80% |
| `Total Spent` | 604 | 4.80% |
| `Discount Applied` | 4,199 | 33.39% |

The largest data-quality issue was the `Discount Applied` column because approximately one-third of its values were missing.

## Data Cleaning Process

### 1. Removing rows with missing item values

Rows with missing values in the `Item` column were removed.

The `Item` column contains categorical information with many distinct values. Since missing item values could not be reliably reconstructed, removing those records was considered more appropriate than assigning an arbitrary category.

This removed **1,213 rows** from the dataset.

### 2. Handling missing numerical values

Missing values in the following numerical columns were replaced using the median:

- `Price Per Unit`
- `Quantity`
- `Total Spent`

The median was selected because it is less affected by extreme values than the mean.

### 3. Removing the `Discount Applied` column

The `Discount Applied` column contained **4,199 missing values**, representing approximately **33.4%** of the original dataset.

Rather than filling a large number of unknown values with the most common value, the column was removed to avoid introducing potentially misleading information into the analysis.

### 4. Checking for duplicates

The dataset contained no duplicate records before cleaning. A duplicate check was also performed after cleaning.

### 5. Data types and categorical variables

The following fields were treated as categorical or text-based variables:

- `Category`
- `Item`
- `Payment Method`
- `Location`

The following fields were treated as identifiers or date-related fields rather than ordinary categorical features:

- `Transaction ID`
- `Customer ID`
- `Transaction Date`

## Outlier Analysis

Potential outliers were examined using two approaches:

### Interquartile Range method

The IQR method identified:

| Numerical column | Potential outliers |
|---|---:|
| `Price Per Unit` | 0 |
| `Quantity` | 0 |
| `Total Spent` | 60 |

The potential outliers in `Total Spent` were retained because high-value transactions may be legitimate and do not necessarily represent errors.

### Z-score method

The Z-score method identified no observations exceeding the selected threshold of 3 standard deviations:

| Numerical column | Potential outliers |
|---|---:|
| `Price Per Unit` | 0 |
| `Quantity` | 0 |
| `Total Spent` | 0 |

### Outlier decision

The potential `Total Spent` outliers were not automatically removed. They were retained for further investigation because they may represent genuine high-value purchases.

This reflects an important data-analysis principle:

> An outlier is not automatically an error.

## Feature Scaling

Two feature-scaling methods were applied to the numerical columns:

- `Price Per Unit`
- `Quantity`
- `Total Spent`

### Standard Scaling

Standard scaling transforms values so that they have an approximate mean of 0 and a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standard = df.copy()

df_standard[numeric_columns] = scaler.fit_transform(
    df_standard[numeric_columns]
)
```

### Min-Max Scaling

Min-Max scaling transforms values to a range between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = df.copy()

df_minmax[numeric_columns] = scaler.fit_transform(
    df_minmax[numeric_columns]
)
```

Standard scaling was considered more suitable for this analysis because the numerical variables have different ranges and units.

## Correlation Analysis

The correlation matrix was used to examine relationships between numerical variables.

The main observations were:

- `Quantity` and `Total Spent` showed a strong positive correlation of approximately **0.712** before cleaning.
- After cleaning, this correlation increased slightly to approximately **0.713**.
- `Price Per Unit` and `Quantity` had an almost zero linear correlation of approximately **0.012**.
- `Price Per Unit` and `Total Spent` showed a positive correlation of approximately **0.631**.
- Cleaning did not significantly change the major relationships between the numerical variables.

The strong relationship between `Quantity` and `Total Spent` is logical because purchasing more units generally increases the total transaction value.

## Before and After Cleaning

| Property | Before cleaning | After cleaning |
|---|---:|---:|
| Number of rows | 12,575 | 11,362 |
| Number of columns | 11 | 10 |
| Missing values | 7,229 | 0 |
| Duplicate records | 0 | 0 |
| Numerical columns | 3 | 3 |
| Potential IQR outliers | 60 | 56 |

The cleaned dataset is saved as:

```text
retail_store_sales_cleaned.csv
```

## Machine-Learning Potential

After preprocessing, the dataset may be suitable for future machine-learning tasks.

### Possible regression problem

A regression model could potentially predict:

```text
Total Spent
```

Possible input features could include:

- `Price Per Unit`
- `Quantity`
- `Category`
- `Payment Method`
- `Location`
- `Transaction Date`

Care must be taken to avoid data leakage. For example, if `Total Spent` is mathematically calculated from `Price Per Unit` and `Quantity`, using both as predictors may produce an overly direct relationship.

### Possible classification problem

A classification model could potentially predict a categorical variable such as:

- `Payment Method`
- `Location`
- `Category`

Before building a model, categorical features would need to be encoded and the target variable would need to be clearly defined.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
retail-store-sales-cleaning-analysis/
│
├── retail_store_sales.csv
├── retail_store_sales_cleaned.csv
├── assignment.ipynb
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone [https://github.com/your-username/retail-store-sales-cleaning-analysis.git](https://github.com/your-username/retail-store-sales-cleaning-analysis.git)
```

Move into the project directory:

```bash
cd retail-store-sales-cleaning-analysis
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run the cells in sequence.

## Key Learning Outcomes

This project demonstrates the following skills:

- Loading CSV data with Pandas
- Inspecting dataset dimensions and data types
- Measuring missing-data percentages
- Handling missing categorical and numerical values
- Removing unreliable columns
- Checking for duplicate records
- Detecting potential outliers
- Comparing IQR and Z-score methods
- Applying StandardScaler and MinMaxScaler
- Creating box plots and correlation heatmaps
- Interpreting relationships between numerical variables
- Preparing data for machine-learning workflows
- Documenting data-cleaning decisions

## Limitations

The cleaned dataset is more suitable for analysis and machine learning, but it should not be considered perfectly error-free.

Potential limitations include:

- Removing rows with missing `Item` values may have removed useful transaction information.
- Removing `Discount Applied` resulted in the loss of discount-related information.
- Retained high-value transactions should be investigated further.
- The dataset does not contain enough business context to explain every sales pattern.
- Additional validation would be required before using the data for production decisions.

## Future Improvements

Possible future improvements include:

- Creating an interactive dashboard using Power BI or Tableau
- Performing time-series analysis of sales trends
- Investigating sales by category and payment method
- Analyzing online versus in-store transactions
- Developing a sales-prediction model
- Creating customer segments
- Investigating the retained high-value transactions
- Adding automated data-quality checks
- Creating a reproducible data-cleaning pipeline

## Author

**Samprit Chakraborty**

B.Tech Computer Science and Engineering Student at KIIT

Interested in:

- Data Analytics
- Data Science
- Artificial Intelligence
- Python
- Machine Learning
- Technology and Business

## License

This project is intended for educational and portfolio purposes.
