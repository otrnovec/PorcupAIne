import os
from flask import Flask, request, render_template, jsonify
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'REST/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Route to render the upload page
@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle file upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    # Check if the user uploaded a file
    if file.filename == '':
        return "No selected file", 400

    # Ensure the uploaded file is a CSV
    if not file.filename.endswith('.csv'):
        return "Invalid file format. Please upload a CSV file.", 400

    # Save the file to the uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the CSV file using pandas
    try:
        df = pd.read_csv(file_path)
        
        # Insert your custom computation here
        result = process_csv(df)
        
        return jsonify(result)
    
    except Exception as e:
        return f"Error processing the file: {str(e)}", 500

def process_csv(df):
    """
    We implement our services here
    """
    return {
        "columns": df.columns.tolist(),
        "row_count": len(df)
    }

if __name__ == '__main__':
    app.run(debug=True)
