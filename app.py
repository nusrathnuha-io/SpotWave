from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import subprocess
import os
import time
app = Flask(__name__, template_folder='templates')

@app.route('/report.txt', methods=['HEAD', 'GET'])
def report():
    if not os.path.exists('report.txt'):
        return "", 404

    if request.method == 'HEAD':
        return "", 200

    return send_from_directory('.', 'report.txt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')  # New route for /index
def index_route():
    return render_template('index.html')

@app.route('/layout1')
def layout1():
    return render_template('layout1.html')

@app.route('/obstacle')
def obstacle():
    return render_template('obstacle.html')

@app.route('/drag')
def drag():
    return render_template('drag.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/view')
def view():
    return render_template('view.html')

@app.route('/save_layout_data', methods=['POST'])
def save_layout_data():
    data = request.get_json()  # Get the JSON data from the request
    app.logger.info("Received data: %s", data)  # Log the received data for debugging
    
    try:
        # Validate input data
        if not data or 'length' not in data or 'width' not in data:
            raise ValueError("Missing length or width in the data.")
        
        length = data['length']
        width = data['width']
        
        # Ensure the new values are logged
        app.logger.info(" %s,  %s", length, width)
        
        # Log the obstacles and walls
        obstacles = data.get('obstacles', [])
        walls = data.get('walls', [])
        
        app.logger.info("Obstacles: %s", obstacles)  # Log obstacles
        app.logger.info("Walls: %s", walls)  # Log walls

        # If validation passes, save the data
        with open('combined_data.txt', 'w') as file:
            # First, write the length and width at the top
            length = data.get('length', 'N/A')
            width = data.get('width', 'N/A')
            file.write(f"{length}\n")
            file.write(f"{width}\n")

            # Write obstacle count
            file.write(f"{len(obstacles)}\n")  # Write the number of obstacles

            # Write obstacles (format: x y material_type)
            for obstacle in obstacles:
                if 'x' in obstacle and 'y' in obstacle and 'type' in obstacle:
                    x = obstacle['x']
                    y = obstacle['y']
                    material_type = obstacle['type']
                    file.write(f"{x} {y} {material_type}\n")
                else:
                    app.logger.error("Invalid obstacle format: %s", obstacle)
            
            # Write walls (format: x1 y1 x2 y2 material_type)
            for wall in walls:
                if all(key in wall for key in ('x1', 'y1', 'x2', 'y2', 'wallType')):
                    x1, y1 = wall['x1'], wall['y1']
                    x2, y2 = wall['x2'], wall['y2']
                    material_type = wall['wallType']
                    file.write(f"{x1} {y1} {x2} {y2} {material_type}\n")
                else:
                    app.logger.error("Invalid wall format: %s", wall)

        return jsonify({"success": True}), 200
    except Exception as e:
        app.logger.error("Error saving layout data: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/obstacle_data', methods=['GET'])
def get_obstacle_data():
    try:
        with open('combined_data.txt', 'r') as file:
            data = file.readlines()
            obstacles = []
            walls = []
            length = None
            width = None

            for line in data:
                line = line.strip()
                if line.startswith("Length:"):
                    length = line.split(": ")[1]
                elif line.startswith("Width:"):
                    width = line.split(": ")[1]
                elif line.startswith("Walls:"):
                    continue  # Skip header line
                elif line.startswith("Obstacles:"):
                    continue  # Skip header line
                else:
                    # Check if line describes a wall or an obstacle
                    if "Type:" in line:  # It's a wall
                        walls.append(line)
                    else:  # It's an obstacle
                        parts = line.split(", ")
                        name = parts[0].split(": ")[1]
                        type_ = parts[1].split(": ")[1]
                        x = float(parts[2].split(": ")[1])
                        y = float(parts[3].split(": ")[1])
                        obstacles.append({'name': name, 'type': type_, 'x': x, 'y': y})

            return jsonify({"length": length, "width": width, "walls": walls, "obstacles": obstacles}), 200
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Combined data file not found."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



@app.route('/analyze_signal_degradation', methods=['POST'])
def analyze_signal_degradation():
    # Redirect to the process page to show loading spinner
    process_page = url_for('process')
    
    # Start the signal degradation script
    subprocess.Popen(['python', 'signal/signal_degradation.py'])  # Modify the path if needed

    # Wait until the signal_degradation.txt file is created in the 'signal' directory
    signal_file_path = 'signal/signal_degradation.txt'  # Update the path as necessary

    # Wait for the signal degradation script to finish and produce the output
    while not os.path.exists(signal_file_path):
        time.sleep(7)  # Wait for 1 second before checking again

    # Now that the signal_degradation.txt file exists, run the placement script
    subprocess.Popen(['python', 'signal/genetic.py'])  # Modify the path if needed

# Wait until the signal_degradation.txt file is created in the 'signal' directory
    signal_file_path = 'report.txt'  # Update the path as necessary

    # Wait for the signal degradation script to finish and produce the output
    while not os.path.exists(signal_file_path):
        time.sleep(7)  # Wait for 1 second before checking again

    return process_page


@app.route('/delete_files', methods=['POST'])
def delete_files():
    try:
        # Define the file paths
        signal_file = 'signal/signal_degradation.txt'
        report_file = 'report.txt'
        
        # Delete the files if they exist
        if os.path.exists(signal_file):
            os.remove(signal_file)
        if os.path.exists(report_file):
            os.remove(report_file)
        
        return jsonify({"message": "Files deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
