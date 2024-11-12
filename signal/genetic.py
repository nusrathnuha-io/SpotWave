import pandas as pd
import random
import re
import os

# Load signal strength data
def load_signal_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'Point \((\d+),\s*(\d+)\): Received Power = ([-\d.]+) dBm', line)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                signal_strength = float(match.group(3))
                data.append({"position": (x, y), "signal_strength": signal_strength})
    return pd.DataFrame(data)

# Initialize population with positions close to the layout center
def initialize_population(data, population_size, num_routers, layout_center, max_distance_from_center=10):
    low_signal_positions = data[data['signal_strength'] < -70]['position'].tolist()
    centralized_positions = [
        pos for pos in low_signal_positions 
        if abs(pos[0] - layout_center[0]) <= max_distance_from_center 
        and abs(pos[1] - layout_center[1]) <= max_distance_from_center
    ]

    population = []
    for _ in range(population_size):
        chromosome = random.sample(centralized_positions, num_routers)
        population.append(chromosome)
    return population

# Fitness function with debugging output
def fitness_function(chromosome, data, min_signal=-70, max_signal=-50, coverage_threshold=0.75):
    covered_points = 0
    total_points = len(data)
    
    for position in chromosome:
        signals = data[data['position'] == position]['signal_strength']
        if not signals.empty:
            signal_strength = signals.values[0]
            if min_signal <= signal_strength <= max_signal:
                covered_points += 1

    coverage = covered_points / total_points
    score = coverage * 100  # Convert to percentage
    
    penalty = len(chromosome) / 10
    final_score = score - penalty
    
    return final_score if coverage >= coverage_threshold else final_score - 100

# Selection (tournament selection)
def selection(population, fitness_scores, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Crossover with check for single-element chromosomes
def crossover(parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1, parent2  # Skip crossover if not enough points
    
    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation (randomly swap router positions)
def mutation(chromosome, low_signal_positions, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.choice(low_signal_positions)
    return chromosome

# Genetic algorithm function with debugging
def genetic_algorithm(data, population_size=50, initial_routers=1, max_routers=10, generations=100, mutation_rate=0.1, coverage_threshold=0.75, layout_center=(0, 0), max_distance_from_center=10):
    num_routers = initial_routers
    best_placement = None
    best_score = -float('inf')
    
    while num_routers <= max_routers:
        population = initialize_population(data, population_size, num_routers, layout_center, max_distance_from_center)
        
        for generation in range(generations):
            fitness_scores = [fitness_function(chromosome, data, coverage_threshold=coverage_threshold) for chromosome in population]
            population = selection(population, fitness_scores)
            
            next_generation = []
            for i in range(0, len(population), 2):
                parent1, parent2 = population[i], population[(i+1) % len(population)]
                child1, child2 = crossover(parent1, parent2)
                next_generation.extend([child1, child2])
            
            low_signal_positions = data[data['signal_strength'] < -70]['position'].tolist()
            next_generation = [mutation(child, low_signal_positions, mutation_rate) for child in next_generation]
            
            population = next_generation
        
        # Evaluate final population fitness
        fitness_scores = [fitness_function(chromosome, data, coverage_threshold=coverage_threshold) for chromosome in population]
        current_best_score = max(fitness_scores)
        current_best_chromosome = population[fitness_scores.index(current_best_score)]
        
        if current_best_score > best_score:
            best_score = current_best_score
            best_placement = current_best_chromosome
            
            if fitness_function(best_placement, data, coverage_threshold=coverage_threshold) >= coverage_threshold * 100:
                break
            
        num_routers += 1

    return best_placement

def main():
  
   
    data = load_signal_data('signal/signal_degradation.txt')
    
    # Calculate layout center based on signal data
    layout_center_x = (data['position'].apply(lambda p: p[0]).max() + data['position'].apply(lambda p: p[0]).min()) // 2
    layout_center_y = (data['position'].apply(lambda p: p[1]).max() + data['position'].apply(lambda p: p[1]).min()) // 2
    layout_center = (layout_center_x, layout_center_y)

    # Genetic algorithm parameters
    population_size = 50
    initial_routers = 1
    max_routers = 4
    generations = 100
    mutation_rate = 0.1
    coverage_threshold = 0.75
    max_distance_from_center = 15  # Adjust as needed

    best_placement = genetic_algorithm(
        data, 
        population_size, 
        initial_routers, 
        max_routers, 
        generations, 
        mutation_rate, 
        coverage_threshold,
        layout_center,
        max_distance_from_center
    )
    
    # Format the output as specified
    num_routers = len(best_placement)
    report_content = f"Optimal Router Positions: ({num_routers}, {best_placement})\n"
    
    # Display in console
    print(report_content)
    
    # Save results to report.txt in specified format
    report_file_path = 'report.txt' 
    with open(report_file_path, 'w') as report_file:
        report_file.write(report_content)

if __name__ == "__main__":
    main()
