import csv
import os

# Define the CSV file
directory = 'DataCollection/raw_data/gestures'
csv_file = os.path.join(directory, 'gesture_data.csv')  # Combine directory and file name

# Initialize CSV file
def initialize_csv():
    """Create or initialize the CSV file for gesture data."""
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Header: Label + 42 columns for 21 landmarks (x, y for each)
            headers = ["Label"] + [f"x{i+1},y{i+1}" for i in range(21)]
            writer.writerow(headers)

# Save gesture data
def save_gesture_data(landmark_row, label):
    """Save gesture data with label into the CSV file."""
    if landmark_row:  # Ensure landmarks are not empty
        row = [label] + landmark_row  # Append label at the start of the row
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
