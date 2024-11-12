import numpy as np
import matplotlib.pyplot as plt
import random

def read_signal_degradation_results(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        first_line = lines[0].strip()
        print(f"First line of the file: {first_line}")  # Debugging print

        first_line_parts = first_line.split()
        # Adjust to handle "Layout Size:" instead of "Layout dimensions:"
        if len(first_line_parts) >= 4 and first_line_parts[0] == 'Layout' and first_line_parts[1] == 'Size:':
            layout_length = int(float(first_line_parts[2]))  # Convert to int after parsing float
            layout_width = int(float(first_line_parts[3]))  # Convert to int after parsing float
            results = np.zeros((layout_length, layout_width))
            
            for line in lines[1:]:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    point = tuple(map(int, parts[0].strip('Point ()').split(',')))
                    received_power = float(parts[1].split('=')[1].strip().split()[0])
                    results[point] = received_power
            return results
        else:
            raise ValueError(f"Incorrect format in the first line of the input file: {first_line}")


# Function to calculate coverage given router positions and signal strength threshold
def calculate_coverage(results, router_positions, min_signal_strength=-70):
    layout_length, layout_width = results.shape
    coverage = np.zeros_like(results, dtype=bool)
    
    # Ensure router_positions is iterable
    if not isinstance(router_positions, list):
        router_positions = [router_positions]
    
    for (x, y) in router_positions:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if 0 <= x + dx < results.shape[0] and 0 <= y + dy < results.shape[1]:
                    if results[x + dx, y + dy] >= min_signal_strength:
                        coverage[x + dx, y + dy] = True
    return np.sum(coverage)



def find_optimal_router_placement(results, num_routers, min_signal_strength=-70, generations=100, population_size=50):
    layout_length, layout_width = results.shape

    def random_position():
        return (np.random.randint(layout_length), np.random.randint(layout_width))

    def fitness(positions):
        return calculate_coverage(results, positions, min_signal_strength)

    def mutate(positions):
        idx = np.random.randint(len(positions))
        positions[idx] = random_position()
        return positions

    def crossover(parent1, parent2):
        cut = np.random.randint(len(parent1))
        child = parent1[:cut] + parent2[cut:]
        return child

    # Initialize population with random positions
    population = [[random_position() for _ in range(num_routers)] for _ in range(population_size)]
    best_solution = max(population, key=fitness)

    for generation in range(generations):
        new_population = []
        for _ in range(population_size):
            # Use random.sample instead of np.random.choice to avoid the error
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            if np.random.rand() < 0.1:  # mutation probability
                child = mutate(child)
            new_population.append(child)
        population = new_population
        best_solution = max(population, key=fitness)

    return best_solution




# Function to visualize the optimal router placement
def visualize_optimal_placement(results, optimal_positions):
    plt.imshow(results, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Signal Strength (dBm)')
    plt.title('Optimal WiFi Router Placement')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    for (x, y) in optimal_positions:
        plt.text(y, x, 'R', ha='center', va='center', color='blue')
    
    plt.show()

# Function to determine the best number of routers for coverage
def determine_best_number_of_routers(results, min_signal_strength=-70, improvement_threshold=0.01, max_generations=100):
    layout_length, layout_width = results.shape
    best_coverage = 0
    best_positions = []
    best_num_routers = 0
    max_coverage = layout_length * layout_width

    for num_routers in range(1, max_coverage + 1):
        positions = find_optimal_router_placement(results, num_routers, min_signal_strength, max_generations)
        coverage = calculate_coverage(results, positions, min_signal_strength)
        improvement = (coverage - best_coverage) / max_coverage

        if improvement > improvement_threshold or best_num_routers == 0:
            best_coverage = coverage
            best_positions = positions
            best_num_routers = num_routers
        else:
            break

        # Break if the whole area is covered
        if best_coverage >= max_coverage:
            break

    return best_num_routers, best_positions  # Return number and positions

def save_report(optimal_positions):
    with open("report.txt", "w") as report_file:
        report_file.write(f"Optimal Router Positions: {optimal_positions}\n")  # More descriptive


def main():
    results_file_name = "signal/signal_degradation.txt"
    min_signal_strength = -70  # Minimum acceptable signal strength in dBm

    results = read_signal_degradation_results(results_file_name)
    best_num_routers, optimal_positions = determine_best_number_of_routers(results, min_signal_strength)

    print(f"Optimal Router Positions: {optimal_positions}")

    # Save results to report.txt
    save_report(optimal_positions)

    visualize_optimal_placement(results, optimal_positions)

if __name__ == "__main__":
    main()
