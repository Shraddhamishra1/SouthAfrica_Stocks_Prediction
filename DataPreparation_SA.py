# Import necessary libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
from pandas.tseries.offsets import MonthEnd

# Define a custom function to generate a range of dates
def daterange(x, y):
    dates = pd.date_range(x, y, freq='d')
    return dates 

# Get the current working directory
directory = os.getcwd()
print(directory)

# Define the folder path where sales data is stored
folder = r"C:\Users\Shraddha.Mishra\OneDrive - Shell\Career_development\Projects\Ultimate Potential South Africa\OneDrive_1_7-25-2024\Sales Data"
# folder = 'Sales Data'  # Alternative folder path (commented out)

# Construct the full path to the data folder
path = os.path.join(directory, folder)
path

# Read all CSV files in the specified folder and concatenate them into a single DataFrame
full_data = pd.DataFrame()
for root, dirs, files in os.walk(path):
    for file in files:
       if file.endswith(".csv"):  # Process only CSV files
           print(os.path.join(root, file))
           df = pd.read_csv(os.path.join(root, file))
           full_data = pd.concat([df, full_data])           

# Print the shape of the concatenated DataFrame
full_data.shape

# Print the data types of each column in the DataFrame
full_data.dtypes

# Convert 'BL Date' column to datetime format
full_data['date'] = pd.to_datetime(full_data['BL Date'], format='%Y%m%d')

# Define a list of specific material numbers to filter the data
material_list = [400003103, 400003118, 400003139, 400003151, 400003154, 400003159, 400006090]

# Filter the DataFrame to include only rows with material numbers in the defined list
full_data = full_data[full_data['Material Number'].isin(material_list)]

# Print the shape of the filtered DataFrame
full_data.shape

# Read external holiday data from an Excel file
folder = 'External Variables'
file = 'SA Holiday Calendar.xlsx'
holidays = pd.read_excel(os.path.join(directory, folder, file))

# Write the holidays data to a CSV file
subfolder = 'EDA'
subfolder_1 = 'External Vars Processed'
filename = 'holidays.csv'
write_path = os.path.join(directory, subfolder, subfolder_1, filename)
holidays.to_csv(write_path)

# Merge the sales data with the holidays data
full_data_timeseries = pd.merge(full_data, holidays, left_on='date', right_on='Holiday', how='left')

# Drop unnecessary columns and handle missing values
full_data_timeseries = full_data_timeseries.drop('Holiday', axis=1)
full_data_timeseries['Holiday Desc'] = full_data_timeseries['Holiday Desc'].fillna('NA')
full_data_timeseries['Holiday_Grouping'] = full_data_timeseries['Holiday_Grouping'].fillna('NA')
full_data_timeseries['Holiday Flag'] = full_data_timeseries['Holiday Flag'].fillna(0)

# Print the first few rows of the updated DataFrame
full_data_timeseries.head()

# Read lockdown data from an Excel file
folder = 'External Variables'
file = 'SA Lockdown.xlsx'
lockdowns = pd.read_excel(os.path.join(directory, folder, file))

# Remove rows with missing 'End Date' and create date ranges for lockdown periods
lockdowns = lockdowns[~lockdowns['End Date'].isna()] 
lockdowns['Date_Range'] = lockdowns.apply(lambda x: daterange(x['Start Date'], x['End Date']), axis=1)

# Expand date ranges into individual rows
lockdowns = lockdowns.explode('Date_Range')
lockdowns = lockdowns[['Province', 'Date_Range']]
lockdowns.drop_duplicates(inplace=True)

# Read province mapping data from an Excel file
folder = 'Mapping Files'
file = 'SA_Site_Mapping.xlsx'
sites = pd.read_excel(os.path.join(directory, folder, file))
sites = sites[sites['Site Status'] != 'Inactive']
sites = sites[['Ship To Party Code', 'Plant Code', 'Province']]
sites.drop_duplicates(inplace=True)

# Write lockdowns and province mapping data to CSV files
filename = 'lockdowns.csv'
write_path = os.path.join(directory, subfolder, subfolder_1, filename)
lockdowns.to_csv(write_path)

filename = 'province.csv'
write_path = os.path.join(directory, subfolder, subfolder_1, filename)
sites.to_csv(write_path)

# Merge the sales data with lockdowns and site information
full_data_timeseries = pd.merge(full_data_timeseries, sites, left_on=['Ship To', 'Receiving Plant Code'], right_on=['Ship To Party Code', 'Plant Code'], how='left')
full_data_timeseries = pd.merge(full_data_timeseries, lockdowns, left_on=['Province', 'date'], right_on=['Province', 'Date_Range'], how='left')

# Drop redundant columns and create a lockdown flag
full_data_timeseries = full_data_timeseries.drop(['Ship To Party Code', 'Plant Code'], axis=1)
full_data_timeseries['LockDown Flag'] = full_data_timeseries['Date_Range'].apply(lambda x: 0 if pd.isnull(x) else 1)
full_data_timeseries = full_data_timeseries.drop('Date_Range', axis=1)

# Print the first few rows of the updated DataFrame
full_data_timeseries.head()

# Read depot mapping data and material mapping data from Excel files
folder = 'Mapping Files'
file = 'SA_Depot_Mapping.xlsx'
depots = pd.read_excel(os.path.join(directory, folder, file))[['Plant', 'Region']]
depots.drop_duplicates(inplace=True)

file = 'SA_Material_Mapping.xlsx'
materials = pd.read_excel(os.path.join(directory, folder, file))[['Material Number', 'Grade Type']]
materials.drop_duplicates(inplace=True)

# Merge the sales data with depot and material information
full_data_timeseries = pd.merge(full_data_timeseries, depots, left_on=['Receiving Plant Code'], right_on=['Plant'], how='left')
full_data_timeseries = full_data_timeseries.drop('Plant', axis=1)
full_data_timeseries = pd.merge(full_data_timeseries, materials, on=['Material Number'], how='left')

# Print the shape of the updated DataFrame
full_data_timeseries.shape

# Read price details from an Excel file and process them
folder = 'External Variables'
file = 'SA Price Information.xlsx'
price = pd.read_excel(os.path.join(directory, folder, file))

# Sort the price data and create end dates for price periods
price.sort_values(by=['Location', 'Grade Type', 'Date'], inplace=True)
price['End Date'] = price.groupby(['Location', 'Grade Type'])['Date'].shift(-1)
max_date = max(price['Date'])
price['End Date'] = price['End Date'].fillna(max_date.date().strftime('%Y-%m-%d'))

# Create date ranges and expand them into individual rows
price['Date_Range'] = price.apply(lambda x: daterange(x['Date'], x['End Date']), axis=1)
price = price.explode('Date_Range')
price = price[price['Date_Range'] < price['End Date']]
price = price[['Location', 'Grade Type', 'Date_Range', 'Fuel Price']]

# Aggregate price data
agg = {      
   'Fuel Price': 'mean'
}
price = price.groupby(['Grade Type', 'Date_Range']).aggregate(agg).reset_index()
price.drop_duplicates(inplace=True)

# Write price data to a CSV file
filename = 'price.csv'
write_path = os.path.join(directory, subfolder, subfolder_1, filename)
price.to_csv(write_path)

# Merge the sales data with price details
full_data_timeseries = pd.merge(full_data_timeseries, price, left_on=['Grade Type', 'date'], right_on=['Grade Type', 'Date_Range'], how='left')
full_data_timeseries = full_data_timeseries.drop('Date_Range', axis=1)

# Print the shape of the updated DataFrame
full_data_timeseries.shape

# Read and process loyalty and marketing program data
agg = {      
   'Estimated Budget in ZAR': 'sum',
   'Campaign Description': 'nunique'
}
folder = 'External Variables'
file = 'SA Retail Loyalty Data.xlsx'
loyalty = pd.read_excel(os.path.join(directory, folder, file))
loyalty = loyalty[['Campaign Description', 'Estimated Budget in ZAR', 'Start Date', 'End Date']]

# Convert data types and handle missing values
loyalty['Estimated Budget in ZAR'] = pd.to_numeric(loyalty['Estimated Budget in ZAR'], errors='coerce')
loyalty['Campaign Description'] = loyalty['Campaign Description'].astype(str)
print("Loyalty data types after conversion:")

loyalty = loyalty.dropna(subset=['Estimated Budget in ZAR'])

# Create date ranges and expand them into individual rows
loyalty['Date_Range'] = loyalty.apply(lambda x: daterange(x['Start Date'], x['End Date']), axis=1)
loyalty = loyalty.explode('Date_Range')
loyalty = loyalty[['Campaign Description', 'Estimated Budget in ZAR', 'Date_Range']]
loyalty = loyalty.groupby('Date_Range').aggregate(agg).reset_index()

# Read marketing program data and process it similarly
agg = {      
   'Estimated Budget in USD': 'sum',
   'Campaign Description': 'nunique'
}
file = 'SA Marketing Programs.xlsx'
marketing = pd.read_excel(os.path.join(directory, folder, file))
marketing = marketing[['Campaign Description', 'Estimated Budget in USD', 'Start Date', 'End Date']]

# Convert data types and handle missing values
marketing['Estimated Budget in USD'] = pd.to_numeric(marketing['Estimated Budget in USD'], errors='coerce')
marketing['Campaign Description'] = marketing['Campaign Description'].astype(str)
print("Marketing data types after conversion:")

marketing = marketing.dropna(subset=['Estimated Budget in USD'])

# Create date ranges and expand them into individual rows
marketing['Date_Range'] = marketing.apply(lambda x: daterange(x['Start Date'], x['End Date']), axis=1)
marketing = marketing.explode('Date_Range')
marketing = marketing[['Campaign Description', 'Estimated Budget in USD', 'Date_Range']]
marketing = marketing.groupby('Date_Range').aggregate(agg).reset_index()

# Write loyalty and marketing data to CSV files
filename = 'loyalty.csv'
write_path = os.path.join(directory, subfolder, subfolder_1, filename)
loyalty.to_csv(write_path)

filename = 'marketing.csv'
write_path = os.path.join(directory, subfolder, subfolder_1, filename)
marketing.to_csv(write_path)

# Merge sales data with loyalty and marketing information
full_data_timeseries = pd.merge(full_data_timeseries, loyalty, left_on='date', right_on='Date_Range', how='left')
full_data_timeseries = pd.merge(full_data_timeseries, marketing, left_on='date', right_on='Date_Range', how='left')

# Print the final DataFrame shape
full_data_timeseries.shape

# Write the final DataFrame to a CSV file
filename = 'full_data_timeseries.csv'
write_path = os.path.join(directory, subfolder, subfolder_1, filename)
full_data_timeseries.to_csv(write_path, index=False)
