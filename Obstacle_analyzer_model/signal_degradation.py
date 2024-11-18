import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings  # Import the warnings module



# Suppress specific sklearn warning
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names")

# Load the gradient boosting model and encoders
model = joblib.load('gradient_boosting_model.pkl')
encoder = joblib.load('material_type_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')


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
    expected_columns = feature_names
    df = pd.DataFrame([feature_values.flatten()], columns=expected_columns)
    prediction = model.predict(df)
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


def analyze_obstacles(obstacles, transmitted_power, frequency, encoder, model):
    # Determine grid size based on obstacle coordinates
    max_x = max([max(obstacle[0], obstacle[1]) for obstacle in obstacles if len(obstacle) == 3] +
                [max(obstacle[0], obstacle[2]) for obstacle in obstacles if len(obstacle) == 5])
    max_y = max([max(obstacle[0], obstacle[1]) for obstacle in obstacles if len(obstacle) == 3] +
                [max(obstacle[1], obstacle[3]) for obstacle in obstacles if len(obstacle) == 5])
    grid_size = max(max_x, max_y) + 1

    environment_map = np.zeros((grid_size, grid_size))
    results = np.zeros((grid_size, grid_size))

    for obstacle in obstacles:
        if len(obstacle) == 3:
            # Point obstacle
            ox, oy, material_type = obstacle
            if ox >= 0 and oy >= 0 and ox < grid_size and oy < grid_size:
                environment_map[ox, oy] = 1
        elif len(obstacle) == 5:
            # Wall obstacle
            x1, y1, x2, y2, material_type = obstacle
            add_wall_to_map(environment_map, x1, y1, x2, y2)

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

                    # Calculate received power
                    received_power += calculate_received_power(transmitted_power, distance, frequency, material_loss)

                elif len(obstacle) == 5:
                    # Wall obstacle
                    x1, y1, x2, y2, material_type = obstacle
                    wall_distance = distance_to_wall(x, y, x1, y1, x2, y2)
                    if wall_distance < 0.01:  # Avoid dividing by zero
                        continue

                    # Predict material loss
                    material_loss = predict_material_loss(wall_distance, material_type)

                    # Calculate received power
                    received_power += calculate_received_power(transmitted_power, wall_distance, frequency,
                                                               material_loss)

            results[x, y] = received_power
    return results, environment_map


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


# Function to save results to a text file
def save_results(results, environment_map, layout_length, layout_width, file_name):
    with open(file_name, 'w') as file:
        file.write(f"Layout Size: {layout_length} {layout_width}\n")
        for point, received_power in np.ndenumerate(results):
            file.write(f"Point {point}: Received Power = {received_power} dBm\n")


# Function to visualize results using a heatmap
def visualize_results(results, environment_map):
    plt.imshow(results, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Signal Strength (dBm)')
    plt.title('WiFi Signal Strength Heatmap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    for (x, y), value in np.ndenumerate(environment_map):
        if value == 1:
            plt.text(y, x, 'X', ha='center', va='center', color='blue')
    plt.show()


def main():
    # Read parameters from text file
    with open("input_parameters.txt", "r") as file:
        lines = file.readlines()
        layout_length = float(lines[0].strip())  # Parse as float
        layout_width = float(lines[1].strip())  # Parse as float
        num_obstacles = int(lines[2].strip())
        obstacles = []
        for line in lines[3:3 + num_obstacles]:
            coords = line.split()
            if len(coords) == 3:
                # Point obstacle
                x, y, material_type = coords
                obstacles.append((int(x), int(y), material_type.strip()))
            elif len(coords) == 5:
                # Wall obstacle
                x1, y1, x2, y2, material_type = coords
                obstacles.append((int(x1), int(y1), int(x2), int(y2), material_type.strip()))

    transmitted_power = 18.0  # Example transmitted power in dBm
    frequency = 2.4e9  # Example frequency in Hz (2.4 GHz)

    # Analyze obstacles and calculate signal degradation for each point in the grid
    results, environment_map = analyze_obstacles(obstacles, transmitted_power, frequency, encoder, model)

    # Save results to a text file
    save_results(results, environment_map, layout_length, layout_width, "signal_degradation4.txt")
    print("Results saved to 'signal_degradation4.txt'")

    # Visualize results
    visualize_results(results, environment_map)


if __name__ == "__main__":
    main()
