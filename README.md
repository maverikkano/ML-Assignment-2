# ML Assignment 2
**Name:** Abhishek B. Patil
**Student ID:** 2026AJ05083

---

## Problem statement

Implement the following classification models using the dataset chosen above. All the 5 ML models have to be implemented on the same dataset.
1. Logistic Regression 
2. Decision Tree Classifier 
3. K-Nearest Neighbor Classifier 
4. Naive Bayes Classifier - Gaussian or Multinomial 
5. Support Vector Machine

---

## Dataset description

**UCI Machine Learning Repository - Bank Marketing** (`bank-additional-full.csv`, dataset ID 222).
Source: <https://archive.ics.uci.edu/dataset/222/bank+marketing>

- **Instances:** 41,188
- **Features:** 21 in the raw file; 20 used after dropping `duration`
- **Target:** `y` - `yes` (4,640, 11.3%) / `no` (36,548, 88.7%); imbalanced
- **Types:** 10 categorical (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`), 11 numeric (`age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`)
- **Missing values:** none (encoded as the literal string `unknown` inside categorical columns)
- **Why `duration` is dropped:** it is the length of the last call and is only known *after* the call completes. Keeping it leaks the target and produces unrealistically high scores.

The dataset is stored at `data/bank-additional/bank-additional-full.csv`.