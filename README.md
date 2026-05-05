# Used Car Price Prediction

Predicting used car prices using a German automotive listings dataset. Built as a learning project to practice end-to-end ML: data cleaning, EDA, feature engineering, and comparing models.

---

## The Dataset

`cars_sampled.csv`: 50,000 rows of used car listings scraped from a German platform. Each row has details like brand, mileage, fuel type, gearbox, vehicle type, engine power, and whether the car has unrepaired damage.

The data is messy. Prices range from €0 to €12 million, registration years go as far back as 1000 AD, and some power values hit 19,000 PS. A big chunk of the work here was just making the data usable.

---

## What I Did

**Cleaning**
- Dropped columns that clearly don't affect price: `name`, `dateCrawled`, `postalCode`, `lastSeen`
- Removed obviously bad rows — cars "registered" in year 1000, prices of €0, engines with 19,000 PS
- Set working ranges: price €100–€150k, registration year 1950–2020, power 10–500 PS
- After filtering, went from ~50k rows to ~43k usable ones

**Feature engineering**
- Created an `Age` column from `yearOfRegistration` + `monthOfRegistration`, then dropped those two
- Ran EDA on each variable to check if it actually correlates with price — dropped `seller`, `offerType`, and `abtest` since they showed no signal
- One-hot encoded all categorical columns

**Missing values — two approaches**
I wasn't sure whether to drop rows with missing values or fill them, so I tried both and compared:
- **Drop:** 32,884 rows remaining
- **Fill:** median for numeric columns, mode for categorical — kept all 42,772 rows

**Modeling**
- Log-transformed price before modeling (the raw distribution is extremely skewed)
- 70/30 train-test split
- Trained Linear Regression and Random Forest on both versions of the data
- Used a mean-prediction baseline to sanity check the models

---

## Results

**With missing rows dropped:**

| Model | R² (test) | RMSE (test) |
|---|---|---|
| Baseline | — | 1.127 |
| Linear Regression | 0.77 | 0.546 |
| Random Forest | 0.85 | 0.436 |

**With missing values filled:**

| Model | R² (test) | RMSE (test) |
|---|---|---|
| Baseline | — | 1.188 |
| Linear Regression | 0.70 | 0.648 |
| Random Forest | 0.83 | 0.494 |

Dropping missing rows gave better results than imputing them. I guess the missingness wasn't random, certain types of listings tend to omit fields like `notRepairedDamage`, so filling those with the mode probably introduced noise rather than signal.

Random Forest outperformed Linear Regression by a decent margin, which makes sense — car pricing isn't really a linear problem. A 10-year-old BMW and a 10-year-old Fiat shouldn't be treated the same way just because their age is identical.

The three features most correlated with price were engine power (r=0.58), mileage (r=-0.44), and age (r=-0.34).

---

## Stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib

---

## Run it

```bash
git clone https://github.com/dinakarbl/Used-car-price-prediction.git
cd Used-car-price-prediction
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
jupyter notebook "Used car price prediction.ipynb"
```
