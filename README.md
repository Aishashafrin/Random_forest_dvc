# Random Forest – End-to-End ML Pipeline using DVC

## Project Overview

This project implements an end-to-end machine learning pipeline using the Breast Cancer Wisconsin dataset from scikit-learn. It uses a Random Forest Classifier to classify tumors and DVC to manage pipeline dependencies, reproducibility and experiment tracking.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Git and DVC
- Pickle

## Project Workflow

1. Load and validate the raw dataset.
2. Split the data into 80% training and 20% testing sets.
3. Train a Random Forest Classifier using hyperparameters from `params.yaml`.
4. Evaluate the model using accuracy, precision, recall, F1-score and ROC-AUC.
5. Extract and save the top 10 important features.
6. Generate a confusion matrix and error analysis report.
7. Use DVC to reproduce the pipeline and compare experiment metrics.

## Project Outputs

- `model.pkl` — trained Random Forest model
- `metrics.json` — model evaluation metrics
- `feature_importance.csv` — top 10 important features
- `confusion_matrix.png` — confusion matrix visualization
- `error_analysis.txt` — classification error analysis

## Installation

```bash
pip install -r requirements.txt
```

## Run the Pipeline

```bash
dvc repro
```

## View Metrics

```bash
dvc metrics show
dvc metrics diff
```

## Experiment Tracking

Change the hyperparameters in `params.yaml` and execute `dvc repro` again. Use `dvc metrics show` and `dvc metrics diff` to compare the results.

## Conclusion

This project demonstrates reproducible machine learning, DVC pipeline management, hyperparameter experimentation, feature importance analysis and error analysis using Random Forest.

# What I Learned from Random Forest + DVC Pipeline

Through this project, I learned how to develop a complete and reproducible machine learning pipeline using Random Forest and DVC.

**1. Data loading and validation**

I learned how to load the Breast Cancer Wisconsin dataset, validate the data and save it as CSV for further processing.

**2. Data preprocessing**

I learned how to split the dataset into training and testing sets using an 80/20 ratio. Random Forest does not generally require feature scaling because its decision trees split features using thresholds rather than distance calculations.

**3. Random Forest model training**

I learned how to train a Random Forest Classifier and configure hyperparameters such as `n_estimators`, `max_depth`, `min_samples_split` and `random_state` through `params.yaml`.

**4. DVC pipeline management**

I learned how to create pipeline stages, define dependencies and reproduce the complete workflow using `dvc repro`. DVC reruns stages when relevant dependencies or commands change.

**5. Experiment tracking**

I learned to compare evaluation metrics after changing hyperparameters using `dvc metrics show` and `dvc metrics diff`.

**6. Model evaluation**

I learned to evaluate classification performance using accuracy, precision, recall, F1-score and ROC-AUC.

**7. Feature importance**

I learned how Random Forest estimates feature importance and how to save the ten most important features to a CSV file. I also learned that impurity-based importance can be biased.

**8. Error analysis**

I learned how to interpret a confusion matrix, identify false positives and false negatives, and suggest improvements based on classification errors.

**Conclusion**

This project improved my understanding of the end-to-end machine learning workflow, experiment tracking, version control and reproducibility. I also learned the importance of evaluating model performance and documenting results systematically.
