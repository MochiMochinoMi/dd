# Function to stack the data from each sheet
import pandas as pd
import os 
import zipfile
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib
import xlrd
import matplotlib.patches as mpatches

matplotlib.use('Agg')

def extract_zip(zip_path):
    # Create a temporary folder to extract the files
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_folder)

    return temp_folder
from datetime import datetime, timedelta

def avail_stocks(path, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date() 
    df = pd.read_csv(path,index_col="Date") 
    df.index = pd.to_datetime(df.index).date
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    if df.empty:
        return None
    else:
        L = list(np.unique(np.array(df["Name"])))
        return L




def avail_stocksd(path,selected_date):
  selected_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
  df = pd.read_csv(path,index_col="Date") 
  df.index = pd.to_datetime(df.index).date
  df = df[(df.index == selected_date)]
  if df.empty:
        return None
  else:
    L = list(np.unique(np.array(df["Name"])))
    return L

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime
def stack_excel_data(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Create an empty DataFrame to store the stacked data
    stacked_df = pd.DataFrame(columns=['Name', 'Signal', 'Prediction', 'Duration', 'Date'])

    # Iterate over each file
    for file in files:
        # Check if the file is an Excel file
        if file.endswith('.xlsx') or file.endswith('.xls'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file)

            # Load the Excel sheet into a DataFrame
            df_sheet1 = pd.read_excel(file_path, sheet_name=0)
            df_sheet2 = pd.read_excel(file_path, sheet_name=1)
            date_str = df_sheet1.columns[8]
            date = pd.to_datetime(date_str, format='%d_%b_%Y')

            # Function to stack the data from each sheet
            def stack_data(df, stacked_df, date):
                # Iterate through the desired column ranges
                column_ranges = [(1, 6), (7, 12), (13, 18)]
                for start_col, end_col in column_ranges:
                    # Extract the relevant columns
                    part_df = df.iloc[:, start_col:end_col]

                    # Create a temporary DataFrame to hold the stacked data for this iteration
                    temp_df = pd.DataFrame(columns=['Name', 'Signal', 'Prediction'])

                    # Iterate over the rows starting from row index 4
                    i = 4
                    while i < part_df.shape[0]:
                        if part_df.iloc[i].isnull().any():
                            i += 3  # Skip the current row and the next two rows
                        else:
                            # Extract the values for each column in the row
                            values_name = part_df.iloc[i].values
                            values_signal = part_df.iloc[i + 1].values
                            values_prediction = part_df.iloc[i + 2].values

                            # Append the values to the temporary DataFrame
                            temp_df = pd.concat([temp_df, pd.DataFrame({
                                'Name': values_name,
                                'Signal': values_signal,
                                'Prediction': values_prediction
                            })], ignore_index=True)

                        i += 3

                    # Reset the index of the temporary DataFrame
                    temp_df = temp_df.reset_index(drop=True)

                    # Extract the string value from the first four rows
                    string_value = part_df.iloc[:4].values.flatten().tolist()
                    string_value = next((str_val for str_val in string_value if isinstance(str_val, str)), None)

                    # Store the string value in a duration variable
                    duration = string_value

                    # Add the duration and date columns to the temporary DataFrame
                    temp_df['Duration'] = duration
                    temp_df['Date'] = date

                    # Append the temporary DataFrame to the stacked DataFrame
                    stacked_df = pd.concat([stacked_df, temp_df], ignore_index=True)

                return stacked_df

            # Stack the data from sheet one
            stacked_df = stack_data(df_sheet1, stacked_df, date)
            # Stack the data from sheet two
            stacked_df = stack_data(df_sheet2, stacked_df, date)

    # Sort the DataFrame by the index (Date) in ascending order
    stacked_df.sort_values('Date', inplace=True)

    # Set the 'Date' column as the index of the DataFrame
    stacked_df.set_index('Date', inplace=True)
    stacked_df['Cumulative Value'] = stacked_df['Signal'] * stacked_df['Prediction']
    stacked_df.to_csv('website/static/data.csv', index=True, date_format='%Y-%m-%d')
    return stacked_df

def plot_stock_data(path, start_date, end_date, duration, stock_names, plot_variable, log_scale=False):
    df = pd.read_csv(path,index_col="Date")     
    df.index = pd.to_datetime(df.index).date
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    filtered_data = df[(df.index >= start_date) & (df.index <= end_date) & (df['Duration'] == duration)]
    
    plt.figure(figsize=(12, 6))
    
    for stock_name in stock_names:
        stock_data = filtered_data[filtered_data['Name'] == stock_name]
        if len(stock_data) > 0:
            plt.plot(stock_data.index, stock_data[plot_variable], marker='o', label=stock_name)
    
    plt.xlabel('Date')
    plt.ylabel(plot_variable)
    plt.title(f'{plot_variable} For the ({duration}) Duration - from {start_date} to {end_date}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if len(stock_names) > 0:
        plt.xlim(start_date, end_date)
    
    if len(filtered_data) == 0:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        duration_days = (end_date - start_date).days
        
        ax = plt.gca()
        if duration_days <= 60:  # Less than or equal to 2 months
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        elif duration_days > 60 and duration_days <= 365:  # Between 2 months and 1 year
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        else:  # Larger than 1 year
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
    if log_scale:
        plt.yscale('log')
    
    if len(filtered_data) == 0 or plot_variable not in filtered_data.columns:
        message = f'No {plot_variable} data available'
        plt.annotate(message, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='center', color='red')
    save_path = "website/static/GRAPH.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')



def plot_single_daily_data(path, ticker, time, log_scale=False):
    data = pd.read_csv(path,index_col="Date") 
    data.index = pd.to_datetime(data.index).date
    time = datetime.strptime(time, "%Y-%m-%d").date()

    # Filter the data for the desired ticker
    filtered_data = data[data['Name'] == ticker]

    # Find the closest available date to the specified time
    closest_time = time

    # Define the desired categories and their order
    categories = [
        'Today to 3 days ahead',
        '7 days',
        '14 days',
        '1 month',
        '3 months',
        'year'
    ]

    # Create bar charts for single daily data set
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    opacity = 0.8
    colors = ['red', 'green', 'blue']  # Assign colors to prediction, cumulative value, and signal

    missing_vars = []  # Track the missing variables for each time duration

    for i, category in enumerate(categories):
        category_data = filtered_data[filtered_data['Duration'] == category]
        
        # Check if any variables are missing in the current time duration
        if category_data.empty or closest_time not in category_data.index:
            missing_vars.append((i, category))
            continue
        
        x = [i + j * bar_width for j in range(3)]  # Three bars for each category

        y = [
            category_data[category_data.index == closest_time]['Prediction'].values[0],
            category_data[category_data.index == closest_time]['Cumulative Value'].values[0],
            category_data[category_data.index == closest_time]['Signal'].values[0]
        ]
        
        plt.bar(x, y, width=bar_width, alpha=opacity, color=colors)

    plt.xlabel('Time Category')
    plt.ylabel('Value')

    # Add the missing variables message to the graph title
    title = f'Single Daily Data Set - Ticker: {ticker} - Time: {closest_time}'
    if missing_vars:
        missing_vars_str = ', '.join(category for _, category in missing_vars)
        title += f'\nVariable(s) Missing for Time Duration(s): {missing_vars_str}'
    plt.title(title)

    plt.xticks([i + bar_width for i in range(len(categories))], categories)

    # Create custom legend
    legend_patches = [
        mpatches.Patch(color='red', label='Prediction'),
        mpatches.Patch(color='green', label='Cumulative Value'),
        mpatches.Patch(color='blue', label='Signal')
    ]
    plt.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()

    if log_scale:
        plt.yscale('log')
    plot_file_path = 'website/static/GRAPH2.png'  # Modify the file path as needed
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    plt.close()






import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_stock_data_2(path, start_date, end_date, stock_name, plot_variable, log_scale=False):
    df = pd.read_csv(path, index_col="Date")
    df.index = pd.to_datetime(df.index).date
    df.sort_index(inplace=True)
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    filtered_data = df[(df.index >= start_date) & (df.index <= end_date) & (df['Name'] == stock_name)]

    plt.figure(figsize=(12, 6))

    unique_dates = filtered_data.index.unique()
    num_unique_dates = len(unique_dates)

    for duration in filtered_data['Duration'].unique():
        duration_data = filtered_data[filtered_data['Duration'] == duration]
        if len(duration_data) > 0:
            duration_data.sort_index(inplace=True)  # Sort data by index
            plt.plot(duration_data.index, duration_data[plot_variable], marker='o', label=duration, linestyle='-')

    plt.xlabel('Date')
    plt.ylabel(plot_variable)
    plt.title(f'{plot_variable} for Stock {stock_name} - from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    duration_days = (end_date - start_date).days

    ax = plt.gca()

    if duration_days <= 60:  # Less than or equal to 2 months
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif duration_days <= 365:  # Less than or equal to 1 year
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:  # Larger than 1 year
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    if len(filtered_data) == 0 or plot_variable not in filtered_data.columns:
        message = f'No {plot_variable} data available for Stock {stock_name}'
        plt.annotate(message, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='center', color='red')

    save_path = "website/static/GRAPHTD.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')










