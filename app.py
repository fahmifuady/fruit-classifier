from flask import Flask, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
from model import predict

# Define the Flask application
app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get the prediction from the model
        prediction = predict(filepath)

        # Convert the file path to use forward slashes and remove the static folder
        image_url = filepath.replace("\\", "/").replace("static/", "")

        return render_template('result.html', prediction=prediction, image_url=image_url)

    return redirect(request.url)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
