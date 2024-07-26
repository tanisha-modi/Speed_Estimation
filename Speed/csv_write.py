import csv

# Example set
my_set = {"apple", "banana", "cherry", "date"}
my_set1 = {"apple", "banana", "cherry", "date", "hello"}

# Convert set to list to maintain order (if order matters)
my_list = list(my_set)
my_list1 = list(my_set1)

# Define the CSV file path
csv_file_path = "output.csv"

# Write the list to a CSV file, each element in a separate column
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(my_list)
    writer.writerow(my_list1)

print(f"Set elements saved to {csv_file_path} successfully.")