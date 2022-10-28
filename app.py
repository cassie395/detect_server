from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
from matplotlib.pyplot import box
import numpy as np 
import cv2
import base64
from yolo import runModel

app = Flask(__name__)

@app.route('/detectObject' , methods=['POST'])
def detect_image():

	file = request.files['image'].read()
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

	box_num = runModel(img)
	#print(box_num)

	return jsonify({'status':str(box_num)})


@app.route('/manual' , methods=['GET', 'POST'])
def manual():
	box_num = [0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0]
	#print(box_num)

	return jsonify({'status':str(box_num)})

	
@app.after_request
def after_request(response):
    print("[log] setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=5000, debug = False)
