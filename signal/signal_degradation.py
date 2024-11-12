import math
import numpy as np
import pandas as pd
import joblib
import os

# Load the gradient boosting model and encoders with correct paths
model = joblib.load('signal/gradient_boosting_model.pkl')
encoder = joblib.load('signal/material_type_encoder.pkl')
feature_names = joblib.load('signal/feature_names.pkl')

# Function to calculate Free Space Path Loss (FSPL)
def calculate_fspl(distance, frequency):
    c = 3e8  # Speed of light in meters per second
    return 20 * math.log10(distance) + 20 * math.log10(frequency) + 20 * math.log10(4 * math.pi / c)

# Function to calculate Distance Loss using the inverse square law
def calculate_distance_loss(distance):
    return 10 * math.log10(distance)

def predict_material_loss(distance, material_type):
    features = pd.DataFrame([[distance, material_type]], columns=['distance', 'material_type'])
    encoded_features = encoder.transform(features[['material_type']])
    if isinstance(encoded_features, np.ndarray):
        feature_values = np.hstack((features.drop('material_type', axis=1).values, encoded_features))
    else:
        feature_values = np.hstack((features.drop('material_type', axis=1).values, encoded_features.toarray()))

    # Convert feature_values to numpy array without column names
    prediction = model.predict(feature_values)
    return prediction[0]


# Function to calculate received signal power based on transmitted signal power and losses
def calculate_received_power(transmitted_power, distance, frequency, material_loss):
    fspl = calculate_fspl(distance, frequency)
    distance_loss = calculate_distance_loss(distance)
    return transmitted_power - (fspl + material_loss + distance_loss)

# Function to add a wall to the environment map
def add_wall_to_map(environment_map, x1, y1, x2, y2, wall_thickness=1):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if x1 >= 0 and y1 >= 0 and x1 < environment_map.shape[0] and y1 < environment_map.shape[1]:
            environment_map[x1, y1] = 1  # Mark the wall on the map
        if (x1 == x2) and (y1 == y2):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def analyze_signal_strength(transmitted_power, frequency, encoder, model, grid_size, obstacles):
    # Initialize a dictionary to store results for each router placement
    all_results = {}

    # Iterate over every possible router placement in the grid
    for router_x in range(grid_size):
        for router_y in range(grid_size):
            # Initialize grid for storing received power at each point for this router position
            results = np.zeros((grid_size, grid_size))

            for x in range(grid_size):
                for y in range(grid_size):
                    received_power = transmitted_power
                    for obstacle in obstacles:
                        if len(obstacle) == 3:
                            # Point obstacle
                            obstacle_x, obstacle_y, obstacle_type = obstacle
                            distance = math.sqrt((x - obstacle_x) ** 2 + (y - obstacle_y) ** 2)
                            if distance < 0.01:  # Avoid dividing by zero
                                continue

                            # Predict material loss
                            material_loss = predict_material_loss(distance, obstacle_type)
                            
                            # Calculate received power for the current router position
                            received_power += calculate_received_power(transmitted_power, distance, frequency, material_loss)
                            
                        elif len(obstacle) == 5:
                            # Wall obstacle
                            x1, y1, x2, y2, material_type = obstacle
                            wall_distance = distance_to_wall(x, y, x1, y1, x2, y2)
                            if wall_distance < 0.01:  # Avoid dividing by zero
                                continue
                            
                            # Predict material loss
                            material_loss = predict_material_loss(wall_distance, material_type)
                            
                            # Calculate received power for the current router position
                            received_power += calculate_received_power(transmitted_power, wall_distance, frequency, material_loss)

                    # Calculate distance loss and FSPL for the router to each grid point
                    router_distance = math.sqrt((x - router_x) ** 2 + (y - router_y) ** 2)
                    if router_distance >= 0.01:
                        fspl = calculate_fspl(router_distance, frequency)
                        distance_loss = calculate_distance_loss(router_distance)
                        received_power -= (fspl + distance_loss)

                    # Save the calculated received power
                    results[x, y] = received_power

            # Store the results for this router position
            all_results[(router_x, router_y)] = results

    return all_results

# Function to calculate the distance from a point to a wall segment
def distance_to_wall(px, py, x1, y1, x2, y2):
    if (x1 == x2) and (y1 == y2):
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    line_mag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_mag ** 2
    if u < 0:
        closest_point = (x1, y1)
    elif u > 1:
        closest_point = (x2, y2)
    else:
        closest_point = (x1 + u * (x2 - x1), y1 + u * (y2 - y1))
    return math.sqrt((px - closest_point[0]) ** 2 + (py - closest_point[1]) ** 2)

# Function to save results for each router placement
def save_results_for_all_positions(all_results, file_name):
    with open(file_name, 'w') as file:
        for position, results in all_results.items():
            file.write(f"Router Position: {position}\n")
            for (x, y), received_power in np.ndenumerate(results):
                file.write(f"Point ({x}, {y}): Received Power = {received_power} dBm\n")
            file.write("\n")

# Main function to execute the analysis and save the results
def main():
    
    
        
    # Read parameters from text file
    with open("combined_data.txt", "r") as file:
        lines = file.readlines()
        layout_length = float(lines[0].strip())  # Parse as float
        layout_width = float(lines[1].strip())   # Parse as float
        num_obstacles = int(lines[2].strip())
        obstacles = []
        for line in lines[3:3+num_obstacles]:
            coords = line.split()
            if len(coords) == 3:
                # Point obstacle
                x, y, material_type = coords
                obstacles.append((int(x), int(y), material_type.strip()))
            elif len(coords) == 5:
                # Wall obstacle
                x1, y1, x2, y2, material_type = coords
                obstacles.append((int(x1), int(y1), int(x2), int(y2), material_type.strip()))
    
    grid_size = int(max(layout_length, layout_width))
    transmitted_power = 18.0  # Example transmitted power in dBm
    frequency = 2.4e9  # Example frequency in Hz (2.4 GHz)

    # Analyze signal strength for each router position
    all_results = analyze_signal_strength(transmitted_power, frequency, encoder, model, grid_size, obstacles)

    # Save all results to a text file
    save_results_for_all_positions(all_results, "signal/signal_degradation.txt")
    
    

if __name__ == "__main__":
    main()
