# Predictive Analysis of Police Offense Data

## Project Goal
This project focuses on analyzing police offense data to build predictive models for two primary targets:
1.  **Offense Status (`offensestatus`):** To predict the final status of a reported police offense (e.g., Cleared by Arrest, Suspended, Open).
2.  **Offense Description (`offensedescription`):** To predict the type of offense, after reducing the high cardinality of original descriptions using fuzzy matching techniques.

The project involves data cleaning, exploratory data analysis (EDA) to uncover patterns in crime data, and the development and evaluation of several machine learning models.

## Dataset
The primary dataset used is `Police_Bulk_Data_2014_20241027.csv`, which contains records of police offenses.

## Data Cleaning & Preprocessing
The data underwent several cleaning and preprocessing steps:
1.  **Missing Value Handling:**
    * Columns with over 90% missing values were dropped (e.g., `offensesignal2`, `offensebusinessblock`).
    * Remaining missing numerical values were imputed using the median of their respective columns.
    * Remaining missing categorical values were imputed using the mode of their respective columns.
2.  **Outlier Treatment:**
    * Outliers in numerical columns (`offensereportingarea`, `offensezip`, `offensepropertyattackcode`) were detected using the Interquartile Range (IQR) method and removed.
3.  **Date/Time Feature Engineering:**
    * Date columns (`offensedate`, `offensereporteddate`) were converted to datetime objects.
    * Time features such as `offense_hour` (from `offensestarttime`), `offense_month`, and `offense_day_of_week` (from `offensedate`) were extracted to aid in temporal analysis and modeling.
4.  **Categorical Feature Encoding:**
    * `LabelEncoder` was used for encoding categorical features in the base model for `offensestatus` and for the target variable `offensestatus` itself.
    * `OneHotEncoder` was used for categorical features when modeling `offensedescription`.
5.  **Handling Class Imbalance:**
    * For the `offensestatus` prediction, `SMOTE` (Synthetic Minority Over-sampling Technique) was applied to address class imbalance in the improved model.
    * For the `offensedescription` prediction, `RandomOverSampler` was used in the improved model.
6.  **Target Variable Reduction (`offensedescription`):**
    * The `offensedescription` column, having a high number of unique values (7892 initially), was processed using `rapidfuzz` library to group similar offense descriptions based on token sort ratio, thereby reducing its cardinality for more effective modeling.
    * Rare categories in other categorical columns were combined into an 'Other' category.
7.  **Data Downsampling:**
    * For modeling `offensedescription`, the dataset was downsampled to 10% of its original size to manage memory usage during one-hot encoding and model training.

## Exploratory Data Analysis (EDA)
EDA was performed to understand the underlying patterns in the crime data:
* **High-Crime Areas:** Identified top 10 offense beats with the highest number of reported incidents (e.g., beat 318 had 895 incidents).
* **Temporal Crime Patterns:**
    * Analyzed crime distribution by the day of the week, showing Friday as the day with the most incidents (10407).
    * Analyzed crime distribution by the hour of the day, with peak hours observed around 5 PM (17:00).
* **Common Crime Types:** Identified the top 10 most common offense descriptions (e.g., CRIMINAL MISCHIEF was the most common with 5075 incidents).
* **Spatial-Temporal Combination:** Visualized crime distribution by day and hour for the top 5 high-crime beats using heatmaps to identify specific spatio-temporal hotspots.

Visualizations included bar plots, line plots, and heatmaps.

## Predictive Modeling

Two main prediction tasks were undertaken:

### 1. Predicting Offense Status (`offensestatus`)

* **Base Model:**
    * **Algorithm:** `RandomForestClassifier` with `class_weight='balanced'`.
    * **Preprocessing:** Datetime and object columns were dropped, and remaining categorical features were LabelEncoded.
    * **Result:** Achieved an accuracy of approximately 66.73%.
* **Improved Model (with SMOTE and XGBoost):**
    * **Algorithm:** `XGBClassifier` (utilizing GPU with `tree_method='hist'`, `device='cuda'`).
    * **Preprocessing:** Similar to the base model but included `SMOTE` for handling class imbalance and `SimpleImputer` for any remaining missing values.
    * **Result:** Achieved an accuracy of approximately 83.66%.
    * Feature importance and confusion matrices were plotted for both models. Error analysis was performed for the XGBoost model.

### 2. Predicting Offense Description (`reduced_offensedescription`)

* **Base Model (Downsampled Data):**
    * **Algorithm:** `LogisticRegression`.
    * **Preprocessing:** Data downsampled (10%), `offensedescription` reduced using fuzzy matching, categorical features one-hot encoded, and feature selection performed using `SelectFromModel` with a `RandomForestClassifier`.
    * **Result:** Achieved an accuracy of approximately 9.80% (focus on top 10 classes by support due to high cardinality).
* **Improved Model (Downsampled Data with Oversampling and Hyperparameter Tuning):**
    * **Algorithm:** `RandomForestClassifier`.
    * **Preprocessing:** Similar to the base `offensedescription` model but included `RandomOverSampler` for class imbalance (after train-test split) and `GridSearchCV` for hyperparameter tuning of the RandomForest model. Feature selection was done using `SelectFromModel` with a `RandomForestClassifier`.
    * **Result:** Achieved an accuracy of approximately 85.88% (after removing rare classes with < 5 samples and focusing on the remaining classes).
    * Metrics visualized included accuracy, precision/recall for top 10 classes, and F1-score for top 10 classes. A macro-average ROC curve was also plotted.

## Evaluation Metrics
The models were evaluated using:
* Accuracy
* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix
* Feature Importance
* ROC Curve and AUC (mentioned as applicable for binary classification)
* Precision-Recall Curve (mentioned as applicable for binary classification)

## Results Summary
* **Offense Status Prediction:** The improved XGBoost model significantly outperformed the base RandomForest model, achieving an accuracy of ~83.66% after addressing class imbalance with SMOTE and utilizing GPU acceleration.
* **Offense Description Prediction:** The improved RandomForest model, after downsampling, fuzzy matching for target reduction, oversampling, and hyperparameter tuning, achieved an accuracy of ~85.88% on the valid classes.

## Libraries Used
* `pandas`
* `numpy`
* `matplotlib.pyplot`
* `seaborn`
* `sklearn` (for model selection, preprocessing, ensemble methods, metrics, linear models, feature selection, imputation)
* `scipy.sparse` (for `hstack`, `csr_matrix`)
* `rapidfuzz` (for fuzzy string matching)
* `imblearn` (for oversampling techniques like SMOTE and RandomOverSampler)
* `xgboost`
* `collections.Counter`
* `tabulate`

## Setup and Installation
1.  Ensure Python 3 is installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy rapidfuzz imbalanced-learn xgboost tabulate jupyterlab
    ```
    The notebook specifically installs `rapidfuzz`:
    ```python
    !pip install rapidfuzz
    ```
3.  The notebook utilizes Google Colab and mounts Google Drive to access the dataset:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    The dataset path is specified as:
    ```python
    file_path = '/content/drive/MyDrive/ADTA_Final_Project/Police_Bulk_Data_2014_20241027.csv'
    ```

## How to Run
1.  Upload the Jupyter Notebook (`ProjectNotebook_Team3.ipynb`) to Google Colab or a local Jupyter environment.
2.  Ensure the dataset (`Police_Bulk_Data_2014_20241027.csv`) is accessible at the path specified in the notebook, or modify the `file_path` variable accordingly. If using Google Colab, place the dataset in your Google Drive as per the path.
3.  Run the notebook cells sequentially.
4.  For GPU acceleration with XGBoost, ensure a GPU runtime is selected in Google Colab (Runtime > Change runtime type > Hardware accelerator > GPU).

## File Structure
* `ProjectNotebook_Team3.ipynb`: The main Jupyter Notebook containing all the code for data processing, analysis, and modeling.
* `Police_Bulk_Data_2014_20241027.csv`: The dataset file (assumed to be in the specified Google Drive path).# Predictive-Crime-Analysis
