from PIL import Image
import numpy as np
import csv
import pandas as pd

# Load the image using PIL
image = Image.open('test.jpg').convert('L')

# Convert the image to a NumPy array
image_array = np.array(image)

# Generate column names based on the width of the image
column_names = [f'Pixel_{i+1}' for i in range(image_array.shape[1])]

# Write the NumPy array to a CSV file with column names
csv_path = 'image.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(column_names)  # Write column names
    for row in image_array:
        writer.writerow(row)

# Load the CSV file into a DataFrame and display the first few rows
df = pd.read_csv(csv_path)
print(df.head())