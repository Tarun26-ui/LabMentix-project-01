# Lab-Mentix-
# Zomato Restaurant Analysis – Hyderabad

An end-to-end ML project on Zomato restaurant data from Hyderabad covering exploratory analysis, sentiment analysis on customer reviews, and rating prediction using machine learning.

---

## What this project is about

I was curious about what actually makes people give a restaurant a high rating on Zomato — is it the cuisine, the price point, how many collections it's listed in, or something in the tone of the review itself? This project tries to answer that by combining restaurant metadata with ~10,000 customer reviews and then building a model to predict whether a restaurant will be rated 4 or above.

The project has four main parts:
1. **Data Wrangling** – cleaning messy cost strings, fixing the rating column (which had a "Like" category mixed in with numeric values), and merging both datasets
2. **EDA** – 15+ visualizations across univariate, bivariate, and multivariate analysis
3. **Sentiment Analysis** – using TextBlob to extract polarity scores from review text
4. **ML Modeling** – Logistic Regression, Random Forest, and Gradient Boosting with GridSearchCV tuning and cross-validation

---

## Dataset
| File | Description | Size |
|------|-------------|------|
| `Zomato_Restaurant_names_and_Metadata.csv` | 105 restaurants with name, cost, cuisines, collections, timings, and Zomato link | 105 rows × 6 cols |
| `Zomato_Restaurant_reviews.csv` | Customer reviews scraped from Zomato | ~10,000 rows × 7 cols |



**Metadata columns:** `Name`, `Links`, `Cost`, `Collections`, `Cuisines`, `Timings`

**Reviews columns:** `Restaurant`, `Reviewer`, `Review`, `Rating`, `Metadata`, `Time`, `Pictures`

> Data is for Hyderabad restaurants only. Reviews span multiple years.

---

## Project Structure

```
zomato-restaurant-analysis/
│
├── Zomato_Project.ipynb                      # Main notebook (EDA + NLP + ML)
├── Zomato_Restaurant_names_and_Metadata.csv  # Restaurant metadata
├── Zomato_Restaurant_reviews.csv             # Customer reviews
└── README.md
```

---

## Setup

```bash
git clone https://github.com/tarun26/zomato-restaurant-analysis.git
cd zomato-restaurant-analysis
pip install pandas numpy matplotlib seaborn textblob scikit-learn
```


```bash
jupyter notebook Zomato_Project.ipynb
```
## Key Findings

- **Ratings skew positive** — most reviews cluster around 3.5 and 4.0; very few restaurants consistently get below 3
- **Cost doesn't strongly predict ratings** — some of the most expensive restaurants have average reviews; mid-range places often outperform them
- **Sentiment polarity from TextBlob correlates well with actual ratings** — reviews with positive polarity scores (> 0.2) are significantly more likely to be 4+
- **Collection membership matters** — restaurants listed in Zomato's curated collections (e.g. "Best Buffets", "Great Nightlife") tend to have higher average ratings, likely a reinforcement effect
- **Random Forest gave the best results** after tuning, with accuracy around 78% on the held-out test set and an AUC-ROC of ~0.84

- 
## Models Compared

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | ~72% | ~0.71 | ~0.79 |
| Random Forest | ~78% | ~0.77 | ~0.84 |
| Gradient Boosting | ~76% | ~0.75 | ~0.82 |

GridSearchCV was used for tuning Random Forest and Gradient Boosting. Cross-validation (5-fold) was used to validate generalization.

---

## Tech Stack

- **Python 3**
- pandas, NumPy — data manipulation
- Matplotlib, Seaborn — visualization
- TextBlob — NLP / sentiment analysis
- scikit-learn — ML models, preprocessing, evaluation

---

## Notebook Sections

1. Know Your Data
2. Understanding Your Variables
3. Data Wrangling
4. Data Visualization & EDA (15+ charts)
5. Hypothesis Testing
6. Feature Engineering & Selection
7. ML Modelling (Logistic Regression, Random Forest, Gradient Boosting)
8. Conclusion

