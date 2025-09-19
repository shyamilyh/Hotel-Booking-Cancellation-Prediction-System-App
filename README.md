# Hotel Booking Cancellation Prediction System

## Project overview

This project builds a machine learning system to predict whether a hotel booking will be canceled. It combines exploratory data analysis, feature engineering, classification modeling, model evaluation, and a simple Streamlit-based deployment so the trained model can be used to make real-time predictions.

**Why this matters (business context)**

Hotels lose revenue and face operational inefficiencies when bookings are canceled last-minute. A reliable cancellation-prediction model helps revenue managers and reservation teams to:

* Target communications or incentives to high-risk bookings
* Overbook strategically to maintain occupancy without excessive risk
* Understand factors that contribute to cancellations and adjust policies or pricing

---

## Dataset & preprocessing

* **Source:** The dataset used is the widely-available hotel bookings dataset (included in the notebook as `hotel_bookings.csv`).
* **Rows / splits:** The notebook uses the full dataset and an 80/20 train-test split. After feature selection and encoding, training and test shapes were:

  * `X_train`: (95,512, 18 features)
  * `X_test`:  (23,878, 18 features)

**Key preprocessing steps**

* Parse `arrival_date` and extract components (month, day-of-week, week number).
* Handle categorical variables using one-hot encoding.
* Create engineered features such as `lead_time`, `stays_in_weekend_nights`, `adr` (average daily rate), flags for popular months, and aggregated counts.
* Ensure training/test split with `random_state=42` for reproducibility.

---

## Exploratory Data Analysis (highlights)

* The dataset contains temporal patterns in cancellations — certain months and lead times show higher cancellation rates.
* `adr` (average daily rate) distribution shows considerable spread (min: -6.38, median ≈ 94.58, max ≈ 211.07).
* Class distribution and trends were visualized; plots present monthly cancellation counts and daily cancellation rates over time to show seasonality and spikes.

---

## Models tried

The notebook trains and compares several classification algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors

**Modeling choices**

* One-hot encoding for categorical features.
* Models trained with default/tuned hyperparameters for baseline comparison.
* Randomized search was used to tune the Random Forest hyperparameters (search over `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`), with `n_estimators=50` for computation-speed considerations.

---

## Results (selected)

Model performance was evaluated on the held-out test set using Accuracy, Precision, Recall, F1 score, and confusion matrices.

* **Logistic Regression**

  * Accuracy: **0.7773**
  * Precision: **0.8863**
  * Recall: **0.4673**
  * F1 Score: **0.6119**

* **Decision Tree**

  * Accuracy: **0.8085**
  * Precision: **0.7418**
  * Recall: **0.7520**
  * F1 Score: **0.7469**

* **Tuned Random Forest (final choice in notebook)**

  * Accuracy: **0.8335**
  * Precision: **0.8726**
  * Recall: **0.6521**
  * F1 Score: **0.7464**
  * Confusion matrix (test): `[[14053, 854], [3121, 5850]]`

> The tuned Random Forest produced the best overall performance balancing precision and recall while maintaining higher accuracy.

**Notes on evaluation**

* High precision for Logistic Regression shows it is conservative (fewer false positives) but low recall means many canceled bookings were missed.
* Decision Tree improved recall considerably; Random Forest improved accuracy and kept a good balance between precision and recall.
* Depending on business priorities (minimize false negatives vs. false positives), a different operating point or model could be chosen.

---

## Feature importance & insights

The Random Forest model's feature-importance analysis highlights which variables contribute most to predicting cancellations (the notebook computes and visualizes feature importances). Typical important features include:

* `lead_time` (longer lead times often change cancellation likelihood)
* `adr` (price-related effects)
* `deposit_type` and `previous_cancellations`
* Arrival date features (month, popular season flag)

These insights can guide operational decisions, targeted outreach, and pricing strategies.

---

## Deployment

A Streamlit app is included (`app.py`) that loads the trained model (`best_hotel_cancellation_model.pkl`) and exposes a simple web UI where users can enter booking details and get a cancellation prediction. The notebook also saves the best model using `joblib`.

**How to run locally (quick)**

1. Create a Python environment (recommend Python 3.9+).
2. Install dependencies: `pip install -r requirements.txt` (the core packages the notebook uses are `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`).
3. Place `hotel_bookings.csv` in the project root (or adjust the path in the notebook).
4. Train / re-run the notebook or load the supplied `best_hotel_cancellation_model.pkl`.
5. Run the app: `streamlit run app.py` and open the supplied local URL.

---

## Files included

* `Hotel_Booking_Cancellation_Prediction_System_with_Deployment.ipynb` — full notebook with EDA, modeling, tuning, and deployment code.
* `best_hotel_cancellation_model.pkl` — serialized model saved from the notebook (if present after training).
* `app.py` — Streamlit app for quick predictions.
* `requirements.txt` — list of dependencies.

---

## Reproducibility and notes

* The notebook sets `random_state=42` at multiple steps for reproducibility.
* During deployment, ensure the order and names of features in the input to the model match what the model was trained on — the app includes a note to save and load the training column order to avoid mismatches.

---

## Suggested improvements & next steps

1. **Better class-handling & calibration**: experiment with class-weighted models, SMOTE, or calibrated probability outputs to optimize for a specific business metric (e.g., minimize false negatives).
2. **Feature engineering**: create aggregated features at the customer level (e.g., customer-level historical cancellation rate), incorporate external signals (holidays, events), and test embeddings for high-cardinality categoricals.
3. **Modeling**: try XGBoost / LightGBM with more extensive hyperparameter tuning and cross-validation.
4. **Monitoring & Retraining**: add a retraining schedule and monitoring to detect dataset shift (seasonality changes, pricing changes).
5. **A/B testing**: deploy a model-backed intervention (e.g., targeted offer) and A/B test revenue uplift vs. control group.

---

## Business impact 

Using this model, a hotel/reservations team can proactively reduce revenue loss by identifying bookings with high cancellation probability and applying targeted retention strategies (offers, payment guarantees). With careful threshold selection the model can be tuned to the business’s risk appetite.


