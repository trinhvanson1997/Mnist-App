from flask import Flask
from flask import render_template, request
import base64
from mnist_app.mnist_cnn.predict import predict
import json

app = Flask(__name__)


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def route():
    # Remove some first characters of data
    imgstring =str(request.form['base_64'])
    imgstring = imgstring.replace('data:image/png;base64,', '')
    imgstring = imgstring.replace(' ', '+')

    # Convert base64 image to an image and save it
    imgdata = base64.b64decode(imgstring)
    filename = 'image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()
    number = predict(img_path='image.jpg', checkpoint_path='mnist_cnn/checkpoint')
    print(number)

    return json.dumps({'number': number})


if __name__ == '__main__':
    app.run(port=5000)
