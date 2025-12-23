# QMML Christmas Hackathon - 2nd Place Solution

## Competition Overview
This repository contains my solution for the Queen mary machine learning society Christmas Hackathon, where I achieved **2nd place** in the competition. The challenge was to predict customer return behavior for holiday shopping transactions.

## Problem Statement
Given transaction data from Christmas shopping, predict whether a customer will return a product (`ReturnFlag`). The dataset includes customer demographics, product information, purchase details, and temporal features.

## Dataset
- **Training set**: 8,000 transactions with 25 features
- **Test set**: 2,000 transactions with 24 features
- **Target variable**: ReturnFlag (binary classification)
- **Class balance**: Nearly balanced (~50.46% return rate)

## Solution Approach

### Feature Engineering
The key to achieving 2nd place was strategic feature engineering across three categories:

#### 1. Aggregation Features (3 features)
- **CustomerReturnRate**: Historical return rate per customer (identifies serial returners vs loyal customers)
- **ProductReturnRate**: Historical return rate per product (identifies defective/problematic products)
- **CategoryReturnRate**: Historical return rate per category (captures category-level patterns)

#### 2. Temporal Features (6 features)
- **DayOfWeek**: Day of the week (0-6)
- **Day**: Day of the month
- **Month**: Month of the year
- **IsWeekend**: Binary flag for weekend purchases
- **IsPostChristmas**: Binary flag for post-Christmas purchases (Dec 26+)
- **IsChristmasWeek**: Binary flag for Christmas week (Dec 18+)

#### 3. Original Features (8 features)
- Age
- Quantity
- TotalPrice
- CustomerSatisfaction
- DiscountAmount
- OnlineOrderFlag
- PromotionApplied
- GiftWrap

**Total features used**: 17

### Model
- **Algorithm**: Logistic Regression with Standard Scaling
- **Pipeline**: StandardScaler → LogisticRegression
- **Parameters**: max_iter=3000, random_state=42
- **Training**: Full 8,000 samples for final predictions

### Key Insights
1. **Aggregation features were crucial**: Customer and product-level return patterns provided strong predictive signals
2. **Temporal patterns matter**: Post-Christmas and weekend behavior showed distinct return patterns
3. **Simplicity wins**: A well-engineered feature set with logistic regression outperformed more complex models
4. **Feature selection**: Focused on 17 high-signal features rather than expanding to many weak features

## Results
- **Competition Ranking**: 2nd Place
- **Model Performance**: Optimized through strategic feature engineering and cross-validation

## Repository Structure
```
.
├── qmml_christmas_hackathon.ipynb    # Main solution notebook
├── submission_mohamed_final.csv      # Final predictions
└── README.md                         # This file
```

## How to Run
1. Install required packages:
```bash
pip install pandas numpy scikit-learn
```

2. Ensure data files (train.csv and test.csv) are available in your working directory

3. Run the notebook:
```bash
jupyter notebook qmml_christmas_hackathon.ipynb
```

4. The notebook will generate `submission_mohamed_final.csv` with predictions

## Key Takeaways
- Domain knowledge matters: Understanding holiday shopping patterns informed temporal feature creation
- Target leakage prevention: Used train set statistics for test set aggregations
- Less is more: 17 carefully chosen features beat 55+ one-hot encoded features from baseline
- Return prediction requires understanding both customer behavior and product quality

## Technologies Used
- Python 3.x
- pandas
- NumPy
- scikit-learn (LogisticRegression, StandardScaler, Pipeline)
