import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_path = "data/california_housing.csv"
df = pd.read_csv(data_path)

# Split data
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging
mlflow.set_experiment("California-Housing-Regression")

for model_name, model in [
    ("LinearRegression", LinearRegression()),
    ("DecisionTreeRegressor", DecisionTreeRegressor(max_depth=5))
]:
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Log params and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")
