import face_verify
from face_verify import faceRecognition
import os
import register_face
from flask import Flask, render_template, request
import face_recognition
from register_face import register_face

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])

def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'./to_detect', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = faceRecognition(input_folder_path="./ImageFace",
                                  source="1",
                                  input_file_path=file_path,
                                  output_folder_path="./static/face_detected")
        print(product)
        
    return render_template('predict.html', product = product, user_image = filename)            #file_path can or may used at the place of filename

@app.route("/register", methods = ['GET','POST'])
def register():
    if request.method == 'POST':
        file = request.files['file']
        name = request.form['name']
        filename = file.filename
        file_path = os.path.join(r'./to_register', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        result = register_face(file_path,name)
        return "Entry done "
    else:
        return render_template("register.html")
if __name__ == "__main__":
    app.run(debug=True)