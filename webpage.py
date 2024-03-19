from flask import Flask, render_template, Response,jsonify,request,session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
import cv2

#import pill detection function from pill.py
from pill import video_detection

#Create an instance of the Flask class
app = Flask(__name__, template_folder='web', static_folder='images')
app.config['SECRET_KEY'] = 'key'
app.config['UPLOAD_FOLDER'] = 'videos'

latest_detection_status = "No pills detected"

#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):

    #store uploaded video file
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x = ''):
    global latest_detection_status
    yolo_output = video_detection(path_x)
    latest_detection_status = "No pills detected"
    for detection_, classStrings_ in yolo_output:
        if len(classStrings_)>0:
            latest_detection_status = "Pills Detected " + str(classStrings_)
        ref,buffer=cv2.imencode('.jpg',detection_)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    global latest_detection_status
    yolo_output = video_detection(path_x)
    latest_detection_status = "No pills detected"
    for detection_, classStrings_ in yolo_output:
        if len(classStrings_)>0:
            latest_detection_status = "Pills Detected " + str(classStrings_)
        ref,buffer=cv2.imencode('.jpg',detection_)
 
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

#create app route for the home page
@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('homepage.html')

#create app route for the webcam page
@app.route("/webcam", methods=['GET','POST'])

def webcam():
    session.clear()
    return render_template('cam.html')
@app.route('/FrontPage', methods=['GET','POST'])
def front():
    #create instance for upload file form
    form = UploadFileForm()
    if form.validate_on_submit():
        #save uploaded file
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('video.html', form=form)
@app.route('/video')
def video():
    #return response to display the output video
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

#display webcam video on webcam page
@app.route('/webapp')
def webapp():
    #return Response to display the output video
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

#return detection status
@app.route('/detection_status')
def detection_status():
    global latest_detection_status
    return jsonify(message=latest_detection_status)

#run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)