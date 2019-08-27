import os
import json
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from model.predict import predict_image as model_predict
import io


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = './uploaded_images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the POST request has the file part
        if 'file' not in request.files:
            print('Error: file not uploaded')
            return redirect('/error')
        
        file = request.files['file']

        # check if user has not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('Error: file empty')
            return redirect('/error')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            image_pred, confidence = model_predict(file)
            messages = {
                'file': 'uploads/' + filename,
                'image_pred': str(image_pred),
                'confidence': str(confidence)
            }
            
            messages = json.dumps(messages)
            return redirect(url_for('file_predict', messages=messages))
        else:
            return redirect('/error')

    return render_template('index.html')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/predict')
def file_predict():
    messages = request.args['messages']
    messages = json.loads(messages)

    return render_template('predict.html', predicted_value=messages['image_pred'], 
        predicted_confidence=messages['confidence'],
        image=messages['file'])


############## test app ##############
# @app.route('/prediction')
# def prediction():
#     # ensure an image was properly uploaded to our endpoint
#     if request.method == "POST":
#         file = request.files['file']
#         filename = secure_filename(file.filename)

#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


#         image_pred, confidence = model_predict(file)
#     return image_pred, confidence
######################################


@app.errorhandler(Exception)
def handle_error(e):
    return render_template('error.html')

if __name__ == '__main__':
    app.run(port=5002, debug=True) #host= "0.0.0.0")