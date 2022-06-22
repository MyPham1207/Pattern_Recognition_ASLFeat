#!/usr/bin/env python3
"""
Copyright 2020, Zixin Luo, HKUST.
Image matching example.
"""
import yaml
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

from flask import Flask, flash, request, redirect, url_for, render_template

from utils.opencvhelper import MatcherWrapper
import image_matching as im


UPLOAD_FOLDER = 'static/uploaded'

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
	return render_template('index.html')

file_names = []

@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    filenames = []
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filenames.append(filename)
            file_names.append('static/uploaded/' + filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    print(file_names)

    with open('configs/matching_eval.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    # load testing images.
    rgb_list, gray_list = im.load_imgs(file_names, config['net']['max_dim'])     
    # extract regional features.
    descs, kpts = im.extract_local_features(gray_list, config['model_path'], config['net'])
    # feature matching and draw matches.
    matcher = MatcherWrapper()
    match, mask = matcher.get_matches(
        descs[0], descs[1], kpts[0], kpts[1],
        ratio=config['match']['ratio_test'], cross_check=config['match']['cross_check'],
        err_thld=3, ransac=True, info='ASLFeat')
    # draw matches
    disp = matcher.draw_matches(rgb_list[0], kpts[0], rgb_list[1], kpts[1], match, mask)

    output_name = 'result.jpg'
    plt.imsave('static/output/' + output_name, disp)

    return render_template("index.html", filenames=filenames, result='static/output/result.jpg')

    # return render_template('index.html', filenames=filenames)

# @app.route("/")
# def processing(): 
#     with open('configs/matching_eval.yaml', 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#         print(config)

#     # load testing images.
#     rgb_list, gray_list = im.load_imgs(file_names, config['net']['max_dim'])     
#     # extract regional features.
#     descs, kpts = im.extract_local_features(gray_list, config['model_path'], config['net'])
#     # feature matching and draw matches.
#     matcher = MatcherWrapper()
#     match, mask = matcher.get_matches(
#         descs[0], descs[1], kpts[0], kpts[1],
#         ratio=config['match']['ratio_test'], cross_check=config['match']['cross_check'],
#         err_thld=3, ransac=True, info='ASLFeat')
#     # draw matches
#     disp = matcher.draw_matches(rgb_list[0], kpts[0], rgb_list[1], kpts[1], match, mask)

#     output_name = 'disp.jpg'
#     plt.imsave(output_name, disp)

#     return render_template("index.html", result=output_name)



@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploaded/' + filename), code=301)

app.run()
