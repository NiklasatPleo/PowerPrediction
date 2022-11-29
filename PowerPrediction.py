import mlflow.tracking
import mlflow.pyfunc

from influxdb import InfluxDBClient # install via "pip install influxdb"
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

### Get data and prepare dataframe

client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
client.switch_database('orkney')

def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index
    return df

# Get the last 90 days of power generation data
generation = client.query(
    "SELECT * FROM Generation where time > now()-90d"
    ) # Query written in InfluxQL

# Get the last 90 days of weather forecasts with the shortest lead time
wind  = client.query(
    "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL


gen_df_copy = get_df(generation)
wind_df_copy = get_df(wind)
gen_df = gen_df_copy.copy()
wind_df = wind_df_copy.copy()

# Select relevant columns and join dataframes with inner join
gen_df = gen_df.reset_index()[['time','Total']]
wind_df = wind_df.reset_index()[['time','Direction','Speed']]
df = pd.merge(wind_df,gen_df,on='time')

### Train-Test-Split

# Train set = first 70% of datapoints, Test set = last 30% of datapoints
train_df = df[df['time'] <= df.iloc[round(df.shape[0] * 0.7),:][0]].reset_index(drop=True)
test_df = df[df['time'] > df.iloc[round(df.shape[0] * 0.7),:][0]].reset_index(drop=True)
X_train = train_df[['Direction','Speed']] 
y_train = train_df['Total']
X_test = test_df[['Direction','Speed']] 
y_test = test_df['Total']

### Pipeline

## Prepare pipeline objects
# One-hot-encoding of wind direction
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
ct = ColumnTransformer([("encoder transformer", enc, ["Direction"])], remainder="passthrough") 
# Scaling via MinMaxScaler
sc = MinMaxScaler()
# Modelling
lr_model = LinearRegression()

## Define the pipeline steps 
pipeline = Pipeline(steps=[
                            ("ct", ct),         # One-hot encoding
                            ("sc", sc),         # Scaling
                            ("lr", lr_model)    # Fitting linear model
                        ])


### Evaluation

# Fit pipeline and predict power generation for X_test
pipeline_model = pipeline.fit(X_train, y_train)
y_pred = pipeline_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE is {mse:.1f}, RMSE is {rmse:.1f}")


mlflow.log_metric("mse", mse)
mlflow.log_metric("rmse", rmse)


#lr_model = pipeline_model.named_steps['lr']
#mlflow.pyfunc.save_model("lr_model", python_model=lr_model, conda_env="conda.yaml")

mlflow.sklearn.log_model(pipeline_model, "LinearRegressionModel_logged")
mlflow.sklearn.save_model(pipeline_model, "LinearRegressionModel_saved")


# mlflow.pyfunc.save_model("pipeline_model", python_model=pipeline_model, conda_env="conda.yaml")
# this does not work 
# -> mlflow.exceptions.MlflowException: `python_model` must be a subclass of `PythonModel`. Instead, found an object of type: <class 'sklearn.pipeline.Pipeline'>

# Plotting y_pred vs. y_test
plt.plot(y_test, label = "y_test")
plt.plot(y_pred, label = "y_pred")
plt.legend()
plt.show()

