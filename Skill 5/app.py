from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from result import get_diagnosis

app = Flask(__name__)

# Define the upload folder (make sure it exists)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png'}


# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return redirect(request.url)

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform any processing on the image if needed
        diagnosis = get_diagnosis(filename)
        return render_template('result.html', diagnosis=diagnosis, img=filename)

    return 'Error! Invalid file format.'


if __name__ == '__main__':
    # Ensure the 'uploads' folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    app.run(debug=True)
