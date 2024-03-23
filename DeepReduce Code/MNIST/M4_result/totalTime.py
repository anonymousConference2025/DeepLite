# Function to read execution time from a file
def read_execution_time(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "Execution time:" in line:
                    # Extract and return the time as a float
                    return float(line.split()[-2])
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
    return 0.0

# List of file paths
file_paths = [
    "execution_time_1con.txt",
    "execution_time_2process.txt",
    "execution_time_3predict.txt",
    "execution_time_4last_output.txt",
    "execution_time_hsg.txt",
    "execution_time_lenet.txt"
]

# Calculate the total execution time by reading each file
total_time = sum(read_execution_time(file_path) for file_path in file_paths)

print(f"Total Execution Time: {total_time} seconds")
