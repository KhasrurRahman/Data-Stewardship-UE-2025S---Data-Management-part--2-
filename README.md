# CO₂ Emission Prediction Using Machine Learning  
Data Stewardship Exercise 2 – Group 46 (TU Wien)

In this assignment, I explore a structured dataset of global CO₂ emissions with a focus on European countries. The goal is to access and process FAIR data using DBRepo, train a predictive machine learning model, and store the results in a reproducible and shareable way.

The entire workflow follows the FAIR data principles — Findability, Accessibility, Interoperability, and Reusability.

---

## 📦 Dataset & Tools

The dataset used in this project was originally based on the [Our World in Data CO₂ dataset](https://github.com/owid/co2-data) and accessed via [DBRepo](https://test.dbrepo.tuwien.ac.at) through the REST API.

I used the following tools and libraries:
- Python 3.9+
- `pandas`, `numpy` for data processing
- `scikit-learn` for modeling (Random Forest)
- `matplotlib` and `seaborn` for plotting
- `dbrepo-rest` to access subsets via the API

---

## 🛠️ Project Structure

| File                          | Description                                    |
|-------------------------------|------------------------------------------------|
| `Data Stewardship_Data Management.ipynb` | Main Jupyter notebook with all steps |
| `requirements.txt`           | All Python dependencies with versions          |
| `feature_importance.png`     | Plot showing which features influenced CO₂     |
| `val_pred_vs_actual.png`     | Scatter plot: actual vs predicted CO₂          |
| `validation_predictions.csv` | Model output on validation set                 |
| `test_predictions.csv`       | Model output on test set                       |
| `rf_model.joblib`            | Trained Random Forest model (binary)           |

---

## 🧠 Model Summary

I trained a `RandomForestRegressor` using the following features:

```python
['year', 'population', 'gdp', 'cement_co2', 
 'cement_co2_per_capita', 'co2_growth_abs', 'co2_growth_prct']