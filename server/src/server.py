from flask import Flask
from flask import send_file
from flask import request
from flask import flash
from flask import redirect
from flask_cors import CORS
import os
import datetime
import random
from object_detection import ObjectDetection

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "files_reception/"
CALCULATED_FOLDER = "calculated_images/"

# Рандомное название
def alphanumeric(number):
    return ''.join(random.choice('0123456789ABCDEF') for i in range(number))


# В этом методе вызываю метод нейронки для определения монет
@app.route('/send_image', methods=["POST"])
def send_image():
    start_date = datetime.datetime.now()
    print(start_date)

    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        print(file.filename)
        alphanumeric_filename = alphanumeric(8) + ".jpg"
        print(alphanumeric_filename)
        file.save(os.path.join(UPLOAD_FOLDER, alphanumeric_filename))
        return ObjectDetection(UPLOAD_FOLDER, CALCULATED_FOLDER).calculate(start_date, alphanumeric_filename)


@app.route('/get_image/<filename>')
def get_image(filename):
    print(filename)
    return send_file("calculated_images/" + filename, mimetype='image/jpg')


if __name__ == "__main__":
    app.url_map.strict_slashes = False
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
    app.run(host="0.0.0.0", port=5000)
