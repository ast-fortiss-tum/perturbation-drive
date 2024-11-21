import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Function to parse waypoints
def parse_waypoints(waypoints_str):
    waypoints = []
    if waypoints_str:
        for point in waypoints_str.split('@'):
            x, y, z = map(float, point.split(','))
            waypoints.append((x, y, z))
    return waypoints

# Function to process a single JSON file and return a DataFrame
def process_json_file(file_path, folder):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    table = data[0]  # Assuming only one table exists
    scenario = table.get('scenario', {})
    waypoints_str = scenario.get('waypoints', None)
    waypoints = parse_waypoints(waypoints_str) if waypoints_str else []

    df = pd.DataFrame({
        'frames': table['frames'],
        'pos_0': [p[0] for p in table['pos']],
        'pos_1': [p[1] for p in table['pos']],
        'pos_2': [p[2] for p in table['pos']],
        'xte': table['xte'],
        'speeds': table.get('speeds', [0] * len(table['frames'])),
        'actions_0': [a[0][0] for a in table.get('actions', [[0, 0]] * len(table['frames']))], 
        'actions_1': [a[0][1] for a in table.get('actions', [[0, 0]] * len(table['frames']))],
        'pid_actions_0': [a[0][0] for a in table.get('pid_actions', [[0, 0]] * len(table['frames']))], 
        'pid_actions_1': [a[0][1] for a in table.get('pid_actions', [[0, 0]] * len(table['frames']))],
        'isSuccess': table.get('isSuccess', [False] * len(table['frames'])),
        'timeout': table.get('timeout', [False] * len(table['frames'])),
        
    })
    
    # Additional derived columns
    df['avg_xte'] = df['xte'].abs().mean()
    df['avg_speed'] = df['speeds'].mean()
    df['time'] = df['frames'].max() if df['isSuccess'].iloc[0] else 10000

    df['diff_actions_0'] = df['actions_0'] - df['pid_actions_0']
    df['diff_actions_1'] = df['actions_1'] - df['pid_actions_1']

    # Add file details to DataFrame
    file_name = file_path.split('/')[-1].split("_logs_")[0]
    road_name = file_name.split("road_")[1]
    weather_name = file_name.split("udacity")[1].split("road")[0]
    
    df['road_name'] = road_name
    df['weather'] = weather_name
    
    type_test = folder.split("test_")[1].split("_")[0]
    df['type_test'] = type_test
    type_model = folder.split("_")[-1]
    df['type_model'] = type_model
    
    return df


def extract_road_number(road_name):
    match = re.search(r'(\d+)', road_name)
    return int(match.group(1)) if match else float('inf')  # Use 'inf' as a fallback for safety


def plot_xte_distribution(final_data_df, subdirectory):
    unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
    unique_weathers = sorted(final_data_df['weather'].unique())
    plot_positions=[]

    position = 0  # Initial position for bars
    gap = 3  # Gap between different roads
    bar_width = 1  # Width of each individual bar within a group
    
    plot_values_xte, plot_labels, plot_failure = [], [], []
    
    for road in unique_roads:
        for weather in unique_weathers:
            subsubset = final_data_df[(final_data_df['road_name'] == road) & (final_data_df['weather'] == weather)]
            if not subsubset.empty:
                plot_values_xte.append(subsubset['xte'].to_numpy())
                plot_labels.append(f"{road} {weather}")
                plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                plot_positions.append(position)
                position += bar_width
        position += gap  # Add gap after each road type
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['green' if failure == 1 else 'red' for failure in plot_failure]

    for i, (values, label, color) in enumerate(zip(plot_values_xte, plot_labels, colors)):
        ax.boxplot(values, positions=[plot_positions[i]], patch_artist=True, 
                   boxprops=dict(facecolor=color, color=color), 
                   medianprops=dict(color='black'))
    
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(plot_labels, rotation=90)
    
    # Set the color of the x-tick labels based on success/failure
    for tick_label, failure in zip(ax.get_xticklabels(), plot_failure):
        tick_label.set_color('green' if failure == 1 else 'red')
    
    ax.set_title(f'XTE Distribution for {subdirectory}')
    plt.tight_layout()
    plt.show()

# Function to plot average XTE
def plot_average_xte(final_data_df, subdirectory):
    plot_values_xte_avg, plot_labels, plot_failure, plot_positions = [], [], [], []
    
    unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
    unique_weathers = sorted(final_data_df['weather'].unique())
    
    position = 0  # Initial position for bars
    gap = 3  # Gap between different roads
    bar_width = 1  # Width of each individual bar within a group
    
    for road in unique_roads:
        subset = final_data_df[final_data_df['road_name'] == road]
        for weather in unique_weathers:
            subsubset = subset[subset['weather'] == weather]
            if not subsubset.empty:
                plot_values_xte_avg.append(subsubset['xte'].abs().mean())
                plot_labels.append(f"{road} {weather}")
                plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                plot_positions.append(position)
                position += bar_width
        position += gap  # Add gap after each road type

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['green' if failure == 1 else 'red' for failure in plot_failure]
    ax.bar(plot_positions, plot_values_xte_avg, color=colors, width=bar_width)

    ax.set_xticks(plot_positions)
    ax.set_xticklabels(plot_labels, rotation=90)
    
    # Set the color of the x-tick labels based on success/failure
    for tick_label, failure in zip(ax.get_xticklabels(), plot_failure):
        tick_label.set_color('green' if failure == 1 else 'red')
    
    ax.set_title(f'Average XTE for {subdirectory}')
    plt.tight_layout()
    plt.show()

def calculate_sum_isSuccess_false_per_weather_combination(final_data_df, subdirectory_list):
    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]

    # Dictionary to hold the sum of unsuccessful attempts for each weather condition
    sum_isSuccess_false_dict = {}

    # Sort weather types
    unique_weathers = sorted(final_data_df['weather'].unique())

    # Loop through subdirectories and check for isSuccess=False per road/weather combination
    for subdirectory in filtered_types:
        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]
        for weather in unique_weathers:
            subsubset = subset_df[subset_df['weather'] == weather]
            if not subsubset.empty:
                # Group by road to consider each road-weather combination separately
                road_weather_group = subsubset.groupby('road_name')

                # Sum up for the weather: 1 if any isSuccess=False for that road/weather combination
                isSuccess_false_sum = road_weather_group['isSuccess'].apply(lambda x: 1 if not x.all() else 0).sum()

                # Store the result in the dictionary
                if weather not in sum_isSuccess_false_dict:
                    sum_isSuccess_false_dict[weather] = {}
                sum_isSuccess_false_dict[weather][subdirectory] = isSuccess_false_sum

    # Prepare data for the table
    table_data = []
    for weather, isSuccess_false_values in sum_isSuccess_false_dict.items():
        row = [weather]
        for subdirectory in subdirectory_list:
            row.append(isSuccess_false_values.get(subdirectory, 0))  # Default to 0 if no unsuccessful found
        table_data.append(row)

    # Create DataFrame with results
    columns = ['Weather'] + subdirectory_list
    sum_isSuccess_false_df = pd.DataFrame(table_data, columns=columns)
    print(sum_isSuccess_false_df)
    return sum_isSuccess_false_df

def calculate_avg_timeout_per_weather(final_data_df, subdirectory_list):
    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]

    # Dictionary to hold the sum of timeouts for each weather condition
    sum_timeout_dict = {}

    # Sort weather types
    unique_weathers = sorted(final_data_df['weather'].unique())

    # Loop through subdirectories and check for at least one timeout per road/weather combination
    for subdirectory in filtered_types:
        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]
        for weather in unique_weathers:
            subsubset = subset_df[subset_df['weather'] == weather]
            if not subsubset.empty:
                # Group by road to consider each road-weather combination separately
                road_weather_group = subsubset.groupby('road_name')

                # Sum up for the weather: 1 if any timeout=True for that road/weather combination
                timeout_sum = road_weather_group['timeout'].apply(lambda x: 1 if x.any() else 0).sum()

                # Store the result in the dictionary
                if weather not in sum_timeout_dict:
                    sum_timeout_dict[weather] = {}
                sum_timeout_dict[weather][subdirectory] = timeout_sum

    # Prepare data for the table
    table_data = []
    for weather, timeout_values in sum_timeout_dict.items():
        row = [weather]
        for subdirectory in subdirectory_list:
            row.append(timeout_values.get(subdirectory, 0))  # Default to 0 if no timeouts found
        table_data.append(row)

    # Create DataFrame with results
    columns = ['Weather'] + subdirectory_list
    sum_timeout_df = pd.DataFrame(table_data, columns=columns)


    print(sum_timeout_df)
    return sum_timeout_df

# Function to plot max XTE
def plot_max_xte(final_data_df, subdirectory):
    plot_values_xte_max, plot_labels, plot_failure, plot_positions = [], [], [], []
    
    unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
    unique_weathers = sorted(final_data_df['weather'].unique())
    
    position = 0  # Initial position for bars
    gap = 3  # Gap between different roads
    bar_width = 1  # Width of each individual bar within a group
    
    for road in unique_roads:
        subset = final_data_df[final_data_df['road_name'] == road]
        for weather in unique_weathers:
            subsubset = subset[subset['weather'] == weather]
            if not subsubset.empty:
                max_xte = max(abs(subsubset['xte'].max()), abs(subsubset['xte'].min()))
                plot_values_xte_max.append(max_xte)
                plot_labels.append(f"{road} {weather}")
                plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                plot_positions.append(position)
                position += bar_width
        position += gap  # Add gap after each road type

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['green' if failure == 1 else 'red' for failure in plot_failure]
    ax.bar(plot_positions, plot_values_xte_max, color=colors, width=bar_width)

    ax.set_xticks(plot_positions)
    ax.set_xticklabels(plot_labels, rotation=90)
    
    # Set the color of the x-tick labels based on success/failure
    for tick_label, failure in zip(ax.get_xticklabels(), plot_failure):
        tick_label.set_color('green' if failure == 1 else 'red')
    
    ax.set_title(f'Max XTE for {subdirectory}')
    plt.tight_layout()
    plt.show()

# Function to plot time distribution
def plot_time_distribution(final_data_df, subdirectory):
    plot_values_time, plot_labels, plot_failure, plot_positions = [], [], [], []
    
    unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
    unique_weathers = sorted(final_data_df['weather'].unique())
    
    position = 0  # Initial position for bars
    gap = 3  # Gap between different roads
    bar_width = 1  # Width of each individual bar within a group
    
    for road in unique_roads:
        subset = final_data_df[final_data_df['road_name'] == road]
        for weather in unique_weathers:
            subsubset = subset[subset['weather'] == weather]
            if not subsubset.empty:
                time_in_seconds = subsubset['frames'].max() / 20  # Assuming frames to seconds conversion
                plot_values_time.append(time_in_seconds)
                plot_labels.append(f"{road} {weather}")
                plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                plot_positions.append(position)
                position += bar_width
        position += gap  # Add gap after each road type

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['green' if failure == 1 else 'red' for failure in plot_failure]
    ax.bar(plot_positions, plot_values_time, color=colors, width=bar_width)

    ax.set_xticks(plot_positions)
    ax.set_xticklabels(plot_labels, rotation=90)
    
    # Set the color of the x-tick labels based on success/failure
    for tick_label, failure in zip(ax.get_xticklabels(), plot_failure):
        tick_label.set_color('green' if failure == 1 else 'red')
    
    ax.set_title(f'Time Distribution for {subdirectory}')
    plt.tight_layout()
    plt.show()


def plot_max_xte_subplots(final_data_df, subdirectory_list):
    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]
    n_subplots = len(filtered_types) + 1  # Number of subdirectories + 1 for the difference plot

    # Create subplots: n_subplots rows, 1 column, with shared x-axis
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 5 * n_subplots), sharex=True)  # Share x-axis

    # If there's only one subplot, axes will not be an array, so we need to handle that case
    if n_subplots == 1:
        axes = [axes]

    # Dictionary to hold max_xte for each road-weather combination for comparison
    max_xte_dict = {}

    # Loop to plot 
    for idx, subdirectory in enumerate(filtered_types):
        plot_values_xte_max, plot_labels, plot_failure, plot_positions = [], [], [], []

        # Sort roads based on the number extracted from road names
        unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
        unique_weathers = sorted(final_data_df['weather'].unique())

        position = 0  # Initial position for bars
        gap = 2  # Gap between different roads
        bar_width = 1  # Width of each individual bar within a group

        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]

        for road in unique_roads:
            subset = subset_df[subset_df['road_name'] == road]
            for weather in unique_weathers:
                subsubset = subset[subset['weather'] == weather]
                if not subsubset.empty:
                    max_xte = max(abs(subsubset['xte'].max()), abs(subsubset['xte'].min()))
                    plot_values_xte_max.append(max_xte)
                    plot_labels.append(f"{road} {weather}")
                    plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                    plot_positions.append(position + 0.4)
                    position += bar_width + 0.4

                    # Store the max_xte value for comparison 
                    if (road, weather) not in max_xte_dict:
                        max_xte_dict[(road, weather)] = {}
                    max_xte_dict[(road, weather)][subdirectory] = max_xte

            position += gap  # Add gap after each road type

        # Plot in each subplot
        ax = axes[idx]
        colors = ['green' if failure == 1 else 'red' for failure in plot_failure]
        ax.bar(plot_positions, plot_values_xte_max, color=colors, width=bar_width)

        ax.set_xticks(plot_positions)
        ax.set_xticklabels(plot_labels, rotation=90)

        # Set y-axis limit
        ax.set_ylim(0, 4.2)

        # Title for each subplot
        ax.set_title(f'Max XTE for {subdirectory}')

    # Third plot: Difference 
    diff_plot_values = []
    plot_positions_diff = plot_positions.copy()  # Use the same positions as the other plots for consistency
    plot_labels_diff = plot_labels.copy()  # Use the same labels

    # Now calculate the difference for each road-weather combination
    for road, weather in max_xte_dict:
        if subdirectory_list[0] in max_xte_dict[(road, weather)] and subdirectory_list[1] in max_xte_dict[(road, weather)]:
            diff_xte = max_xte_dict[(road, weather)][subdirectory_list[1]] - max_xte_dict[(road, weather)][subdirectory_list[0]]
            diff_plot_values.append(diff_xte)

    # Plot in the third subplot
    ax_diff = axes[-1]

    ax_diff.bar(plot_positions_diff, diff_plot_values, color='blue', width=bar_width)

    ax_diff.set_xticks(plot_positions_diff)
    ax_diff.set_xticklabels(plot_labels_diff, rotation=90)

    # Set y-axis limit for difference plot
    ax_diff.set_ylim(min(diff_plot_values) * 1.1, max(diff_plot_values) * 1.1)

    # Title for difference plot
    ax_diff.set_title(f'Absolute Difference in Max XTE between {subdirectory_list[0]} and {subdirectory_list[1]}')

    # Add a shared X-axis label for all subplots using fig.text()
    fig.text(0.5, 0.04, 'Road and Weather Conditions', ha='center', fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the shared x-label
    plt.show()

import pandas as pd

import pandas as pd

import pandas as pd

def calculate_avg_xte_per_weather(final_data_df, subdirectory_list):
    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]

    # Dictionary to hold average XTE for each weather condition
    avg_xte_dict = {}

    # Sort weather types
    unique_weathers = sorted(final_data_df['weather'].unique())

    # Loop through subdirectories and calculate the average XTE across all roads for each weather
    for subdirectory in filtered_types:
        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]
        for weather in unique_weathers:
            subsubset = subset_df[subset_df['weather'] == weather]
            if not subsubset.empty:  # Corrected the empty check here
                avg_xte = abs(subsubset['xte'].mean())
                if weather not in avg_xte_dict:
                    avg_xte_dict[weather] = {}
                avg_xte_dict[weather][subdirectory] = avg_xte

    # Prepare data for the table
    table_data = []
    for weather, xte_values in avg_xte_dict.items():
        row = [weather]
        for subdirectory in subdirectory_list:
            row.append(xte_values.get(subdirectory, None))
        if len(subdirectory_list) == 2:
            if subdirectory_list[0] in xte_values and subdirectory_list[1] in xte_values:
                diff_xte = xte_values[subdirectory_list[1]] - xte_values[subdirectory_list[0]]
                row.append(diff_xte)
        table_data.append(row)

    # Create DataFrame with results
    columns = ['Weather'] + subdirectory_list + ['Difference'] if len(subdirectory_list) == 2 else []
    avg_xte_df = pd.DataFrame(table_data, columns=columns)
    print(avg_xte_df)
    return avg_xte_df

import pandas as pd

def calculate_derivative_xte_per_weather(final_data_df, subdirectory_list):
    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]

    # Dictionary to hold average derivative of XTE for each weather condition
    avg_derivative_xte_dict = {}

    # Sort weather types
    unique_weathers = sorted(final_data_df['weather'].unique())

    # Loop through subdirectories and calculate the derivative XTE for each road and weather
    for subdirectory in filtered_types:
        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]
        for weather in unique_weathers:
            weather_derivative_xte = []
            for road in subset_df['road_name'].unique():
                subsubset = subset_df[(subset_df['road_name'] == road) & (subset_df['weather'] == weather)]
                # Only consider rows where the isSuccess is True
                # subsubset_success = subsubset[subsubset['isSuccess'] == True]
                if True:  # Only proceed if there is data and successful scenarios
                    # subsubset['isSuccess'].values[0] and not subsubset['timeout'].values[0]
                    if len(subsubset) > 1:  # Ensure there are at least two points for diff calculation
                        # Calculate the derivative (difference between consecutive xte values)
                        xte_diffs = subsubset['xte'].abs().diff().dropna().abs()  # Calculate the difference between consecutive values
                        avg_xte_diff = xte_diffs.max()
                        weather_derivative_xte.append((avg_xte_diff)/4.5)

            # Calculate the average of derivative XTEs across all roads for the current weather
            if weather_derivative_xte:
                avg_derivative_xte = sum(weather_derivative_xte) / len(weather_derivative_xte)*100
                if weather not in avg_derivative_xte_dict:
                    avg_derivative_xte_dict[weather] = {}
                avg_derivative_xte_dict[weather][subdirectory] = avg_derivative_xte

    # Prepare data for the table
    table_data = []
    for weather, xte_values in avg_derivative_xte_dict.items():
        row = [weather]
        for subdirectory in subdirectory_list:
            row.append(xte_values.get(subdirectory, None))
        table_data.append(row)

    # Create DataFrame with results
    columns = ['Weather'] + subdirectory_list
    derivative_xte_df = pd.DataFrame(table_data, columns=columns)
    print('XTE DERIVATIVE')
    print(derivative_xte_df)
    
    # Return the DataFrame with results
    return derivative_xte_df

def calculate_avg_max_xte_per_weather(final_data_df, subdirectory_list):
    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]

    # Dictionary to hold max XTE for each weather condition
    avg_max_xte_dict = {}

    # Sort weather types
    unique_weathers = sorted(final_data_df['weather'].unique())

    # Loop through subdirectories and calculate the max XTE for each road and weather
    for subdirectory in filtered_types:
        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]
        for weather in unique_weathers:
            weather_max_xte = []
            for road in subset_df['road_name'].unique():
                subsubset = subset_df[(subset_df['road_name'] == road) & (subset_df['weather'] == weather)]
                if not subsubset.empty:  # Only proceed if there is data
                    max_xte = max(abs(subsubset['xte'].max()), abs(subsubset['xte'].min()))
                    weather_max_xte.append(max_xte)

            # Calculate the average of maximum XTEs across all roads for the current weather
            if weather_max_xte:
                avg_max_xte = sum(weather_max_xte) / len(weather_max_xte)
                if weather not in avg_max_xte_dict:
                    avg_max_xte_dict[weather] = {}
                avg_max_xte_dict[weather][subdirectory] = avg_max_xte

    # Prepare data for the table
    table_data = []
    for weather, xte_values in avg_max_xte_dict.items():
        row = [weather]
        for subdirectory in subdirectory_list:
            row.append(xte_values.get(subdirectory, None))
        table_data.append(row)

    # Create DataFrame with results
    columns = ['Weather'] + subdirectory_list
    avg_max_xte_df = pd.DataFrame(table_data, columns=columns)
    print(avg_max_xte_df)
    # Display the DataFrame
    return avg_max_xte_df

def plot_avg_xte_subplots(final_data_df, subdirectory_list):
    

    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]
    n_subplots = len(filtered_types) + 1  # Number of subdirectories + 1 for the difference plot

    # Create subplots: n_subplots rows, 1 column, with shared x-axis
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 5 * n_subplots), sharex=True)  # Share x-axis

    # If there's only one subplot, axes will not be an array, so we need to handle that case
    if n_subplots == 1:
        axes = [axes]

    # Dictionary to hold max_xte for each road-weather combination for comparison
    avg_xte_dict = {}

    # Loop to plot 
    for idx, subdirectory in enumerate(filtered_types):
        plot_values_xte_avg, plot_labels, plot_failure, plot_positions = [], [], [], []

        # Sort roads based on the number extracted from road names
        unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
        unique_weathers = sorted(final_data_df['weather'].unique())

        position = 0  # Initial position for bars
        gap = 2  # Gap between different roads
        bar_width = 1  # Width of each individual bar within a group

        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]

        for road in unique_roads:
            subset = subset_df[subset_df['road_name'] == road]
            for weather in unique_weathers:
                subsubset = subset[subset['weather'] == weather]
                if not subsubset.empty:
                    avg_xte = abs(subsubset['xte'].mean())
                    plot_values_xte_avg.append(avg_xte)
                    plot_labels.append(f"{road} {weather}")
                    plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                    plot_positions.append(position + 0.4)
                    position += bar_width + 0.4

                    # Store the max_xte value for comparison
                    if (road, weather) not in avg_xte_dict:
                        avg_xte_dict[(road, weather)] = {}
                    avg_xte_dict[(road, weather)][subdirectory] = avg_xte

            position += gap  # Add gap after each road type

        # Plot in each subplot
        ax = axes[idx]
        colors = ['green' if failure == 1 else 'red' for failure in plot_failure]
        ax.bar(plot_positions, plot_values_xte_avg, color=colors, width=bar_width)

        ax.set_xticks(plot_positions)
        ax.set_xticklabels(plot_labels, rotation=90)

        # Set y-axis limit
        ax.set_ylim(0, 2)

        # Title for each subplot
        ax.set_title(f'Avg XTE for {subdirectory}')

    # Third plot: Difference 
    diff_plot_values = []
    plot_positions_diff = plot_positions.copy()  # Use the same positions as the other plots for consistency
    plot_labels_diff = plot_labels.copy()  # Use the same labels

    # Now calculate the difference  for each road-weather combination
    for road, weather in avg_xte_dict:
        if subdirectory_list[0] in avg_xte_dict[(road, weather)] and subdirectory_list[1] in avg_xte_dict[(road, weather)]:
            diff_xte = avg_xte_dict[(road, weather)][subdirectory_list[1]] - avg_xte_dict[(road, weather)][subdirectory_list[0]]
            diff_plot_values.append(diff_xte)

    # Plot in the third subplot
    ax_diff = axes[-1]
    ax_diff.bar(plot_positions_diff, diff_plot_values, color='blue', width=bar_width)

    ax_diff.set_xticks(plot_positions_diff)
    ax_diff.set_xticklabels(plot_labels_diff, rotation=90)

    # Set y-axis limit for difference plot
    ax_diff.set_ylim(min(diff_plot_values) * 1.1, max(diff_plot_values) * 1.1)

    # Title for difference plot
    ax_diff.set_title('Absolute Difference in Avg XTE')

    # Add a shared X-axis label for all subplots using fig.text()
    fig.text(0.5, 0.04, 'Road and Weather Conditions', ha='center', fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the shared x-label
    plt.show()

def plot_time_subplots(final_data_df, subdirectory_list):

    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]
    n_subplots = len(filtered_types) + 1  # Number of subdirectories + 1 for the difference plot

    # Create subplots: n_subplots rows, 1 column, with shared x-axis
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 5 * n_subplots), sharex=True)  # Share x-axis

    # If there's only one subplot, axes will not be an array, so we need to handle that case
    if n_subplots == 1:
        axes = [axes]

    # Dictionary to hold max_xte for each road-weather combination for comparison
    avg_time_dict = {}

    # Loop to plot 
    for idx, subdirectory in enumerate(filtered_types):
        plot_values_time, plot_labels, plot_failure, plot_positions = [], [], [], []

        # Sort roads based on the number extracted from road names
        unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
        unique_weathers = sorted(final_data_df['weather'].unique())

        position = 0  # Initial position for bars
        gap = 2  # Gap between different roads
        bar_width = 1  # Width of each individual bar within a group

        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]

        for road in unique_roads:
            subset = subset_df[subset_df['road_name'] == road]
            for weather in unique_weathers:
                subsubset = subset[subset['weather'] == weather]
                if not subsubset.empty:
                    time_in_seconds = subsubset['frames'].max() / 20  # Assuming frames to seconds conversion
                    plot_values_time.append(time_in_seconds)
                    plot_labels.append(f"{road} {weather}")
                    plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                    plot_positions.append(position + 0.4)
                    position += bar_width + 0.4

                    # Store the max_xte value for comparison
                    if (road, weather) not in avg_time_dict:
                        avg_time_dict[(road, weather)] = {}
                    avg_time_dict[(road, weather)][subdirectory] = time_in_seconds

            position += gap  # Add gap after each road type

        # Plot in each subplot
        ax = axes[idx]
        colors = ['green' if failure == 1 else 'red' for failure in plot_failure]
        ax.bar(plot_positions, plot_values_time, color=colors, width=bar_width)

        ax.set_xticks(plot_positions)
        ax.set_xticklabels(plot_labels, rotation=90)

        # Set y-axis limit
        ax.set_ylim(0, 110)

        # Title for each subplot
        ax.set_title(f'Time for {subdirectory}')

    # Third plot: Difference
    diff_plot_values = []
    plot_positions_diff = plot_positions.copy()  # Use the same positions as the other plots for consistency
    plot_labels_diff = plot_labels.copy()  # Use the same labels

    # Now calculate the difference for each road-weather combination
    for road, weather in avg_time_dict:
        if subdirectory_list[0] in avg_time_dict[(road, weather)] and subdirectory_list[1] in avg_time_dict[(road, weather)]:
            diff_xte = avg_time_dict[(road, weather)][subdirectory_list[1]] - avg_time_dict[(road, weather)][subdirectory_list[0]]
            diff_plot_values.append(diff_xte)

    # Plot in the third subplot
    ax_diff = axes[-1]
    ax_diff.bar(plot_positions_diff, diff_plot_values, color='blue', width=bar_width)

    ax_diff.set_xticks(plot_positions_diff)
    ax_diff.set_xticklabels(plot_labels_diff, rotation=90)

    # Set y-axis limit for difference plot
    ax_diff.set_ylim(min(diff_plot_values) * 1.1, max(diff_plot_values) * 1.1)

    # Title for difference plot
    ax_diff.set_title(f'Absolute Difference in Time between {subdirectory_list[0]} and {subdirectory_list[1]}')

    # Add a shared X-axis label for all subplots using fig.text()
    fig.text(0.5, 0.04, 'Road and Weather Conditions', ha='center', fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the shared x-label
    plt.show()

def calculate_avg_time_per_weather(final_data_df, subdirectory_list):
    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]

    # Dictionary to hold average time for each weather condition
    avg_time_dict = {}

    # Sort weather types
    unique_weathers = sorted(final_data_df['weather'].unique())

    # Loop through subdirectories and calculate the average time across all roads for each weather
    for subdirectory in filtered_types:
        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]
        for weather in unique_weathers:
            subsubset = subset_df[subset_df['weather'] == weather]
            if not subsubset.empty:
                time_in_seconds = subsubset['frames'].max() / 20  # Assuming frames to seconds conversion
                if weather not in avg_time_dict:
                    avg_time_dict[weather] = {}
                avg_time_dict[weather][subdirectory] = time_in_seconds

    # Prepare data for the table
    table_data = []
    for weather, time_values in avg_time_dict.items():
        row = [weather]
        for subdirectory in subdirectory_list:
            row.append(time_values.get(subdirectory, None))
        if len(subdirectory_list) == 2:
            if subdirectory_list[0] in time_values and subdirectory_list[1] in time_values:
                diff_time = time_values[subdirectory_list[1]] - time_values[subdirectory_list[0]]
                row.append(diff_time)
        table_data.append(row)

    # Create DataFrame with results
    columns = ['Weather'] + subdirectory_list + ['Difference'] if len(subdirectory_list) == 2 else []
    avg_time_df = pd.DataFrame(table_data, columns=columns)
    print(avg_time_df)
    return avg_time_df

def plot_offline_error_subplots(final_data_df, subdirectory_list):
    

    # Filter subdirectories to those present in subdirectory_list
    filtered_types = [sub for sub in final_data_df['type_model'].unique() if sub in subdirectory_list]
    n_subplots = len(filtered_types) + 1  # Number of subdirectories + 1 for the difference plot

    # Create subplots: n_subplots rows, 1 column, with shared x-axis
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 5 * n_subplots), sharex=True)  # Share x-axis

    # If there's only one subplot, axes will not be an array, so we need to handle that case
    if n_subplots == 1:
        axes = [axes]

    # Dictionary to hold max_xte for each road-weather combination for comparison
    avg_time_dict = {}

    # Loop to plot 
    for idx, subdirectory in enumerate(filtered_types):
        plot_values_time, plot_labels, plot_failure, plot_positions = [], [], [], []

        # Sort roads based on the number extracted from road names
        unique_roads = sorted(final_data_df['road_name'].unique(), key=extract_road_number)
        unique_weathers = sorted(final_data_df['weather'].unique())

        position = 0  # Initial position for bars
        gap = 2  # Gap between different roads
        bar_width = 1  # Width of each individual bar within a group

        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]

        for road in unique_roads:
            subset = subset_df[subset_df['road_name'] == road]
            for weather in unique_weathers:
                subsubset = subset[subset['weather'] == weather]
                if not subsubset.empty:
                    time_in_seconds = abs(subsubset['diff_actions_0'].mean())
                    time_in_seconds2 = abs(subsubset['diff_actions_1'].mean())
                    time_in_seconds = (time_in_seconds+time_in_seconds2)/2
                    plot_values_time.append(time_in_seconds)
                    plot_labels.append(f"{road} {weather}")
                    plot_failure.append(1 if subsubset['isSuccess'].iloc[0] else 0)
                    plot_positions.append(position + 0.4)
                    position += bar_width + 0.4

                    # Store the max_xte value for comparison 
                    if (road, weather) not in avg_time_dict:
                        avg_time_dict[(road, weather)] = {}
                    avg_time_dict[(road, weather)][subdirectory] = time_in_seconds

            position += gap  # Add gap after each road type

        # Plot in each subplot
        ax = axes[idx]
        colors = ['green' if failure == 1 else 'red' for failure in plot_failure]
        ax.bar(plot_positions, plot_values_time, color=colors, width=bar_width)

        ax.set_xticks(plot_positions)
        ax.set_xticklabels(plot_labels, rotation=90)

        # Set y-axis limit
        ax.set_ylim(-1, 1)

        # Title for each subplot
        ax.set_title(f'Time for {subdirectory}')

    # Third plot: Difference 
    diff_plot_values = []
    plot_positions_diff = plot_positions.copy()  # Use the same positions as the other plots for consistency
    plot_labels_diff = plot_labels.copy()  # Use the same labels

    # Now calculate the difference  for each road-weather combination
    for road, weather in avg_time_dict:
        if subdirectory_list[0] in avg_time_dict[(road, weather)] and subdirectory_list[1] in avg_time_dict[(road, weather)]:
            diff_xte = avg_time_dict[(road, weather)][subdirectory_list[1]] - avg_time_dict[(road, weather)][subdirectory_list[0]]
            diff_plot_values.append(diff_xte)

    # Plot in the third subplot
    ax_diff = axes[-1]
    ax_diff.bar(plot_positions_diff, diff_plot_values, color='blue', width=bar_width)

    ax_diff.set_xticks(plot_positions_diff)
    ax_diff.set_xticklabels(plot_labels_diff, rotation=90)

    # Set y-axis limit for difference plot
    ax_diff.set_ylim(min(diff_plot_values) * 1.1, max(diff_plot_values) * 1.1)

    # Title for difference plot
    ax_diff.set_title(f'Absolute Offline error between {subdirectory_list[0]} and {subdirectory_list[1]}')

    # Add a shared X-axis label for all subplots using fig.text()
    fig.text(0.5, 0.04, 'Road and Weather Conditions', ha='center', fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the shared x-label
    plt.show()

# Main function to process all files and create plots
def main():
    parent_directory_path = 'output_logs'
    final_data_df = pd.DataFrame()

    # Iterate over subdirectories and process files
    for subdirectory in os.listdir(parent_directory_path):
        if "perturbation" not in subdirectory and subdirectory.startswith('test'):
            subdirectory_path = os.path.join(parent_directory_path, subdirectory)
            if os.path.isdir(subdirectory_path):
                json_files = [f for f in os.listdir(subdirectory_path) if f.endswith('.json')]
                print(f"Processing {len(json_files)} JSON files in {subdirectory}")
                
                for json_file in json_files:
                    file_path = os.path.join(subdirectory_path, json_file)
                    df = process_json_file(file_path, subdirectory)
                    final_data_df = pd.concat([final_data_df, df], ignore_index=True)

    # Reset index
    final_data_df.reset_index(drop=True, inplace=True)


    # plot_max_xte_subplots(final_data_df,["original","extended5"])
    # plot_avg_xte_subplots(final_data_df,["original","extended5"])
    calculate_avg_max_xte_per_weather(final_data_df,["original","extended5"])
    calculate_derivative_xte_per_weather(final_data_df,["original","extended5"])
    # calculate_avg_xte_per_weather(final_data_df,["original","extended5"])
    calculate_avg_time_per_weather(final_data_df,["original","extended5"])
    calculate_avg_timeout_per_weather(final_data_df,["original","extended5"])
    calculate_sum_isSuccess_false_per_weather_combination(final_data_df,["original","extended5"])
    # plot_time_subplots(final_data_df,["original","extended5"])
    # plot_offline_error_subplots(final_data_df,["original","extended5"])

    unique_types = final_data_df['type_model'].unique()
    for subdirectory in unique_types:
        subset_df = final_data_df[final_data_df['type_model'] == subdirectory]
        plot_xte_distribution(subset_df, subdirectory)
        plot_average_xte(subset_df, subdirectory)
        plot_max_xte(subset_df, subdirectory)
        plot_time_distribution(subset_df, subdirectory)

# Run the main function
if __name__ == "__main__":
    main()