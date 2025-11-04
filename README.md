Machine Failure Prediction using Sensor Data
1. Project Overview

This project implements a predictive maintenance system that identifies and forecasts potential machine failures by analyzing sensor data collected from industrial equipment.
The system leverages data preprocessing, feature engineering, and machine learning (Random Forest Classifier) to predict equipment status as Normal or Failed.
Visualizations such as the Confusion Matrix, ROC Curve, and Feature Importance Plot are used to evaluate performance and interpret model behavior.

2. Objective

To develop a machine learning model that can automatically detect machine health conditions based on multi-sensor readings and predict the likelihood of failure, enabling proactive maintenance and minimizing production downtime.

3. Methodology
a. Data Loading

The dataset (data.csv.csv) is imported using pandas.
The script automatically identifies the target column by scanning for keywords such as:
failure, fail, machine_failure, target, or label.

b. Data Preprocessing

Target Variable Mapping:

Converts categorical or text-based labels (e.g., “Yes”, “Fail”, “Broken”) to binary values (1 = Failure, 0 = Normal).

Numeric targets are thresholded when necessary to maintain binary classification.

Feature Cleaning:

Removes irrelevant columns like id, product_id, or machine_id.

Detects and separates numeric and categorical features.

Pipelines:

Numeric Pipeline: Imputation (Median) + Standard Scaling

Categorical Pipeline: Imputation (Missing values) + One-Hot Encoding

Column Transformation:
Combines both pipelines using ColumnTransformer to ensure consistent preprocessing.

c. Model Development

A Random Forest Classifier is used for prediction due to its robustness and interpretability in handling mixed-type data.
Key configurations:

n_estimators = 200

class_weight = balanced

random_state = 42

n_jobs = -1 (parallel processing for performance)

The dataset is split into training and testing subsets (80/20 ratio) using train_test_split with stratification for balanced target representation.

d. Model Evaluation

Performance metrics are computed using:

Accuracy Score

Classification Report (Precision, Recall, F1-score)

ROC-AUC Score

Confusion Matrix Visualization

Visualizations Generated:

Confusion Matrix Heatmap:
Displays correct vs. incorrect predictions.

ROC Curve:
Plots True Positive Rate vs. False Positive Rate to evaluate model discrimination.

Feature Importance Chart:
Ranks the top 15 sensor features influencing failure prediction.

4. Dataset Description

The dataset must contain columns representing sensor readings and failure labels.
Example structure:

Column Name	Description
machine_id	Unique machine identifier
temperature	Sensor-recorded temperature values
vibration	Vibration intensity readings
pressure	System pressure level
rpm	Machine rotational speed
failure	Target variable (0 = Normal, 1 = Failed)

The script auto-detects column types and applies suitable transformations.

5. Results

After model training, key outputs include:

Model accuracy and classification report printed in the console.

Confusion Matrix and ROC Curve displayed using Matplotlib + Seaborn.

Top 15 most important features plotted to interpret the model.

Trained model saved as model.pkl using Joblib for future inference.

6. Dependencies

This project requires the following Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn joblib

7. Execution Steps

Place your dataset file in the project directory:

C:\Users\Farahan\Downloads\data.csv.csv


Run the script:

python main.py


The script will:

Train and evaluate the model.

Display performance metrics and visualizations.

Save the trained model as model.pkl.

8. Key Insights

The Random Forest model provides high accuracy and robust failure detection across sensor inputs.

Feature importance analysis highlights the most critical sensors contributing to machine health prediction.

The framework supports easy retraining with updated sensor data for continuous performance improvement.

9. Future Enhancements

Integration with IoT platforms for real-time sensor monitoring.

Implementation of Deep Learning models (LSTM, GRU) for time-series prediction.

Automated alert systems for high-risk failure probabilities.

Deployment as a web dashboard or cloud service for industry-scale monitoring.

10. Conclusion

The project successfully demonstrates an end-to-end machine learning pipeline for predicting industrial machine failures using sensor data.
It provides a scalable foundation for predictive maintenance, enabling data-driven decision-making to enhance equipment reliability and operational safety.
