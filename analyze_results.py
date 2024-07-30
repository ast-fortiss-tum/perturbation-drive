import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from perturbationdrive.RoadGenerator.Roads.road_visualizer import visualize_road

# PID function definition
def pid(error, prev_error, velocity, total_error):
    # Constants
    Kp = 0.5
    Kd = 0.0
    Ki = 0.0
    max_steering = 1.0
    max_speed = 12.0

    # Update errors
    diff_err = error - prev_error

    # Calculate steering request
    steering_req = (-Kp * error) - (Kd * diff_err) - (Ki * total_error)
    steering_req = max(-max_steering, min(steering_req, max_steering))

    throttle_val = max(0.2, 1.0 - (Kp * 2 * abs(error)) - (Kd * 10 * abs(diff_err)) - (Ki * 5 * abs(total_error)))
    throttle = throttle_val if velocity < max_speed else 0.0

    # Accumulate total error
    total_error += error

    # Save error for next iteration
    prev_error = error

    return throttle, steering_req, prev_error, total_error

def parse_waypoints(waypoints_str):
    waypoints = []
    if waypoints_str:
        for point in waypoints_str.split('@'):
            x, y, z = map(float, point.split(','))
            waypoints.append((x, y, z))
    return waypoints

# Load the JSON data
file_path = 'logs/donkey_logs_2024-07-29 17:28:55.json'  # Replace with your file path
with open(file_path, 'r') as file:
    data = json.load(file)

# Process each table separately
for idx, table in enumerate(data):
    # Handle cases where 'scenario' key might not be present
    scenario = table.get('scenario', {})
    waypoints = scenario.get('waypoints', [None] * len(table['frames']))
    perturbation_function = scenario.get('perturbation_function', [None] * len(table['frames']))
    perturbation_scale = scenario.get('perturbation_scale', [None] * len(table['frames']))

    # Convert each table to a pandas DataFrame
    df = pd.DataFrame({
        'frames': table['frames'],
        'pos_0': [p[0] for p in table['pos']],
        'pos_1': [p[1] for p in table['pos']],
        'pos_2': [p[2] for p in table['pos']],
        'xte': table['xte'],
        'speeds': table.get('speeds', [0] * len(table['frames'])),  # Fill with 0 if 'speeds' is not present
        'actions_th': [a[1] for a in table.get('actions', [[0, 0]] * len(table['frames']))],
        'actions_st': [a[0] for a in table.get('actions', [[0, 0]] * len(table['frames']))], 
        'scenario_waypoints': waypoints,
        'scenario_perturbation_function': perturbation_function,
        'scenario_perturbation_scale': perturbation_scale
    })

    # Print the columns of the DataFrame
    print(f"Columns of DataFrame {idx}:")
    print(df.columns.tolist())

    # Initialize variables for PID
    prev_error = 0.0
    total_error = 0.0

    # Calculate throttle and steering values based on xte
    throttles = []
    steerings = []

    for index, row in df.iterrows():
        error = row['xte']
        velocity = row['speeds']  # Assuming 'speeds' column represents velocity
        throttle, steering, prev_error, total_error = pid(error, prev_error, velocity, total_error)
        throttles.append(throttle)
        steerings.append(steering)

    # Add throttle and steering values to the DataFrame
    df['pid_th'] = throttles
    df['pid_st'] = steerings

    df['actions_th'] = pd.to_numeric(df['actions_th'], errors='coerce')
    df['pid_th'] = pd.to_numeric(df['pid_th'], errors='coerce')
    df['actions_st'] = pd.to_numeric(df['actions_st'], errors='coerce')
    df['pid_st'] = pd.to_numeric(df['pid_st'], errors='coerce')

    # Calculate MSE for throttle and steering
    df['mse_th'] = (df['actions_th'] - df['pid_th']) ** 2
    df['mse_st'] = (df['actions_st'] - df['pid_st']) ** 2


    # Convert to numpy arrays for plotting
    pos_x_np = df['pos_0'].to_numpy()
    pos_y_np = df['pos_1'].to_numpy()

    waypoints = parse_waypoints(df['scenario_waypoints'][0])
    name=df['scenario_perturbation_function'][0]+" Intensity: "+str(df['scenario_perturbation_scale'][0])
    x = [wp[0] for wp in waypoints]
    y = [wp[2] for wp in waypoints]

    fig, ax = plt.subplots()
    ax.plot(x, y, color="black", linewidth=20, linestyle="-")
    ax.plot(x, y, color="red", linewidth=2, linestyle=":")
    plt.plot(pos_x_np, pos_y_np, label='Trajectory')

    # Set the background color
    ax.set_facecolor("lightgreen")

    # Set the title
    ax.set_title(name, fontsize=10)
    # get range of x and y values
    diff_x = max(x) - min(x)
    diff_y = max(y) - min(y)
    if (diff_x) > (diff_y):
        # x is bigger
        axis_range = range(int(min(x)) - 1, int(max(x)) + 1)
        plt.xticks(axis_range)
        yaxis_range = range(
            int(np.mean(y) - diff_x * 0.5), int(np.mean(y) + diff_x * 0.5)
        )
        plt.yticks(yaxis_range)
    else:
        axis_range = range(int(min(y)) - 1, int(max(y)) + 1)
        plt.yticks(axis_range)
        xaxis_range = range(
            int(np.mean(x) - diff_y * 0.5), int(np.mean(x) + diff_y * 0.5)
        )
        plt.xticks(xaxis_range)

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

    df = df.drop(columns=['scenario_waypoints'])

    # Save the updated DataFrame to a new CSV file
    df.to_csv(f'updated_simulation_data_{idx}.csv', index=False)