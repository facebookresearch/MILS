# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from flask import Flask, render_template_string

# Specify the path to the CSV file directly in the code
file_path = "flux.csv"  # Replace with your actual file path

# Function to rearrange images based on "baseline" and "ours" keywords
def rearrange_images(row):
    if "baseline" in row['image_1'] and "ours" in row['image_2']:
        return row['image_1'], row['image_2'], row['input_text']
    elif "ours" in row['image_1'] and "baseline" in row['image_2']:
        return row['image_2'], row['image_1'], row['input_text']
    else:
        return None, None, row['input_text']  # Handle cases without expected labels

# Load and process the CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path)
    df[['baseline', 'ours', 'input_text']] = df.apply(lambda row: rearrange_images(row), axis=1, result_type="expand")
    df.dropna(subset=['baseline', 'ours'], inplace=True)
    return df[['baseline', 'ours', 'input_text']]

# Initialize the Flask app
app = Flask(__name__)

# Process the CSV file once, outside of any route, so it's static for the app
data = process_csv(file_path)

# HTML template for displaying the images and text prompt in a table
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Comparison - Baseline vs. Ours</title>
</head>
<body>
    <h1>Image Comparison - Baseline vs. Ours</h1>
    <table border="1">
        <tr>
            <th>Text Prompt</th>
            <th>Baseline Image</th>
            <th>Ours Image</th>
        </tr>
        {% for index, row in data.iterrows() %}
        <tr>
            <td>{{ row['input_text'] }}</td>
            <td><img src="{{ row['baseline'] }}" width="200"></td>
            <td><img src="{{ row['ours'] }}" width="200"></td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

# Route to display images
@app.route("/")
def display_images():
    return render_template_string(html_template, data=data)

if __name__ == "__main__":
    app.run(debug=True, port=5010)
