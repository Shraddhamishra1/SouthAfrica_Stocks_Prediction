# %%
#pip install Cmake

# %%
# pip install prophet

# %%
#pip install --upgrade holidays

# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
from prophet import Prophet,plot

# %%
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
    return mape

# %%
directory = os.getcwd()

# %%
folder = 'EDA'
filename = 'ProcessedData.csv'
write_path = os.path.join(directory,folder,filename)

# %%
df = pd.read_csv(os.path.join(write_path))

# %%
agg = {      
    'Sales Volumes in L15': 'sum',
    'Holiday Flag': 'max',
    'Holiday Desc': 'max',
    'LockDown Flag': 'max',
    'Fuel Price': 'mean',
    'Loyalty Flag': 'max',
    'Estimated Budget in ZAR': 'sum',
     'Marketing Flag': 'max',
    'Estimated Budget in USD': 'sum'
}
sales_data = df.groupby(['Material Number','date']).aggregate(agg).reset_index()
sales_data['date'] = pd.to_datetime(sales_data['date'], format='%Y-%m-%d')
sales_data.sort_values(by=['Material Number','date'], inplace=True)

# %%
sales_data['Fuel Price L1'] = sales_data.groupby(['Material Number'])['Fuel Price'].shift(1)
sales_data['Fuel Price PCT Change'] = ((sales_data['Fuel Price']-sales_data['Fuel Price L1'])/sales_data['Fuel Price L1'])*100
sales_data['Fuel Price PCT Change'].fillna(0, inplace = True)

# %%
#Reading the data
train_date = ['2023-11-30','2023-12-31','2024-01-31','2024-02-29']
test_date = ['2023-12-01','2024-01-01','2024-02-01','2024-03-01']
end_date = ['2023-12-31','2024-01-31','2024-02-29','2024-03-17']
periods = [31,31,29,17]
material_list = sales_data['Material Number'].unique()

# %%
#Building Baseline Prophet Model
predicted_df = pd.DataFrame()
for i in range(0,4):
    for x in material_list:
        train_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] <= train_date[i])]
        df = train_data[['date','Sales Volumes in L15']]
        df.rename({'date': 'ds', 'Sales Volumes in L15': 'y'}, axis='columns', inplace = True)

        # Prophet model
        m = Prophet()
        m.fit(df)

        # forecasting
        future = m.make_future_dataframe(periods=periods[i])

        # Predictionns
        forecast = m.predict(future)
        pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        pred = pred[(pred['ds'] >= test_date[i]) & (pred['ds'] <= end_date[i])]
        test_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] >= test_date[i]) & (sales_data['date'] <= end_date[i])]
        pred.reset_index(inplace = True)
        test_data.reset_index(inplace = True)
        forecast = pd.concat([test_data, pred], axis=1)
        predicted_df = pd.concat([predicted_df,forecast], axis = 0)

# %%
folder = 'Modelling Output'
subfolder = 'PROPHET'
filename = 'predictions.csv'
write_path = os.path.join(directory,folder,subfolder,filename)
predicted_df.to_csv(write_path)

# %%
#Building Prophet Model w Holidays
predicted_df = pd.DataFrame()
for i in range(0,4):
    for x in material_list:
        train_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] <= train_date[i])]
        df = train_data[['date','Sales Volumes in L15']]
        df.rename({'date': 'ds', 'Sales Volumes in L15': 'y'}, axis='columns', inplace = True)

        #Adding holidays
        holiday_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Holiday Flag'] == 1)]['date'].unique()
        lockdown_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['LockDown Flag'] == 1)]['date'].unique()
        loyalty_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Loyalty Flag'] == 1)]['date'].unique()
        Marketing_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Marketing Flag'] == 1)]['date'].unique()
    
        public_holidays = pd.DataFrame({
        'holiday': 'public holidays',
        'ds': holiday_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        lockdows = pd.DataFrame({
        'holiday': 'lockdowns',
        'ds': lockdown_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        loyalty = pd.DataFrame({
        'holiday': 'loyalty',
        'ds': loyalty_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        marketing = pd.DataFrame({
        'holiday': 'loyalty',
        'ds': Marketing_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        holidays = pd.concat((public_holidays, lockdows,loyalty,marketing))
    
        #Adding Regressors (Price)
        df['Fuel Price'] = train_data['Fuel Price']
        test_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] >= test_date[i]) & (sales_data['date'] <= end_date[i])]
        m = Prophet(holidays=holidays)
        #m.add_regressor('Fuel Price')
        m.fit(df)
        future = m.make_future_dataframe(periods=periods[i])
        future = future[(future['ds'] >= test_date[i]) & (future['ds'] <= end_date[i])]
        #future = pd.merge(future, test_data[['date','Fuel Price']], left_on = 'ds', right_on = 'date', how= 'left')
        #future.drop('date', inplace = True, axis = 1)
        forecast = m.predict(future)
        pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        pred.reset_index(inplace = True)
        test_data.reset_index(inplace = True)
        forecast = pd.concat([test_data, pred], axis=1)
        predicted_df = pd.concat([predicted_df,forecast], axis = 0)

# %%
folder = 'Modelling Output'
subfolder = 'PROPHET'
filename = 'predictions_w_holidays.csv' 
write_path = os.path.join(directory,folder,subfolder,filename)
predicted_df.to_csv(write_path)

# %%
#Building Prophet Model w Holidays & Regressor (Price)
predicted_df = pd.DataFrame()
for i in range(0,4):
    for x in material_list:
        train_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] <= train_date[i])]
        df = train_data[['date','Sales Volumes in L15']]
        df.rename({'date': 'ds', 'Sales Volumes in L15': 'y'}, axis='columns', inplace = True)

        #Adding holidays
        holiday_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Holiday Flag'] == 1)]['date'].unique()
        lockdown_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['LockDown Flag'] == 1)]['date'].unique()
        loyalty_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Loyalty Flag'] == 1)]['date'].unique()
        Marketing_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Marketing Flag'] == 1)]['date'].unique()
    
        public_holidays = pd.DataFrame({
        'holiday': 'public holidays',
        'ds': holiday_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        lockdows = pd.DataFrame({
        'holiday': 'lockdowns',
        'ds': lockdown_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        loyalty = pd.DataFrame({
        'holiday': 'loyalty',
        'ds': loyalty_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        marketing = pd.DataFrame({
        'holiday': 'loyalty',
        'ds': Marketing_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        holidays = pd.concat((public_holidays, lockdows,loyalty,marketing))
    
        #Adding Regressors (Price)
        df['Fuel Price'] = train_data['Fuel Price']
        test_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] >= test_date[i]) & (sales_data['date'] <= end_date[i])]
        m = Prophet(holidays=holidays)
        m.add_regressor('Fuel Price')
        m.fit(df)
        future = m.make_future_dataframe(periods=periods[i])
        future = future[(future['ds'] >= test_date[i]) & (future['ds'] <= end_date[i])]
        future = pd.merge(future, test_data[['date','Fuel Price']], left_on = 'ds', right_on = 'date', how= 'left')
        future.drop('date', inplace = True, axis = 1)
        forecast = m.predict(future)
        pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        pred.reset_index(inplace = True)
        test_data.reset_index(inplace = True)
        forecast = pd.concat([test_data, pred], axis=1)
        predicted_df = pd.concat([predicted_df,forecast], axis = 0)

# %%
folder = 'Modelling Output'
subfolder = 'PROPHET'
filename = 'predictions_w_price.csv'
write_path = os.path.join(directory,folder,subfolder,filename)
predicted_df.to_csv(write_path)

# %%
#Building Prophet Model w Holidays & Regressor (Price Change)
predicted_df = pd.DataFrame()
for i in range(0,4):
    for x in material_list:
        train_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] <= train_date[i])]
        df = train_data[['date','Sales Volumes in L15']]
        df.rename({'date': 'ds', 'Sales Volumes in L15': 'y'}, axis='columns', inplace = True)

        #Adding holidays
        holiday_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Holiday Flag'] == 1)]['date'].unique()
        lockdown_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['LockDown Flag'] == 1)]['date'].unique()
        loyalty_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Loyalty Flag'] == 1)]['date'].unique()
        Marketing_list = sales_data[(sales_data['Material Number'] == x) & (sales_data['Marketing Flag'] == 1)]['date'].unique()
    
        public_holidays = pd.DataFrame({
        'holiday': 'public holidays',
        'ds': holiday_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        lockdows = pd.DataFrame({
        'holiday': 'lockdowns',
        'ds': lockdown_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        loyalty = pd.DataFrame({
        'holiday': 'loyalty',
        'ds': loyalty_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        marketing = pd.DataFrame({
        'holiday': 'loyalty',
        'ds': Marketing_list,
        'lower_window': 0,
        'upper_window': 1,
        })
        holidays = pd.concat((public_holidays, lockdows,loyalty,marketing))
    
        #Adding Regressors (Price)
        df['Fuel Price PCT Change'] = train_data['Fuel Price PCT Change']
        test_data = sales_data[(sales_data['Material Number'] == x) & (sales_data['date'] >= test_date[i]) & (sales_data['date'] <= end_date[i])]
        m = Prophet(holidays=holidays)
        m.add_regressor('Fuel Price PCT Change')
        m.fit(df)
        future = m.make_future_dataframe(periods=periods[i])
        future = future[(future['ds'] >= test_date[i]) & (future['ds'] <= end_date[i])]
        future = pd.merge(future, test_data[['date','Fuel Price PCT Change']], left_on = 'ds', right_on = 'date', how= 'left')
        future.drop('date', inplace = True, axis = 1)
        forecast = m.predict(future)
        pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        pred.reset_index(inplace = True)
        test_data.reset_index(inplace = True)
        forecast = pd.concat([test_data, pred], axis=1)
        predicted_df = pd.concat([predicted_df,forecast], axis = 0)

# %%
folder = 'Modelling Output'
subfolder = 'PROPHET'
filename = 'predictions_w_price_change.csv'
write_path = os.path.join(directory,folder,subfolder,filename)
predicted_df.to_csv(write_path)