#!/usr/bin/env python

import torch
import torchvision

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 120) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objectCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './videos/fullrez.mp4'
arguments_strFps = '30'
arguments_strBitRate = "12M"

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--fps' and strArgument != '': arguments_strFps = strArgument # allow FPS value selection from cmd line
	if strOption == '--bitrate' and strArgument != '': arguments_strBitRate = strArgument # allow FPS value selection from cmd line
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
	# end

	##########################################################
if __name__ == '__main__':
	numpyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	intWidth = numpyImage.shape[1]
	intHeight = numpyImage.shape[0]

	dblRatio = float(intWidth) / float(intHeight)

	process_load(numpyImage, {})

	objectFrom = {
	'dblCenterU': intWidth / 2.0, #You can change these but many settings will crash it
	'dblCenterV': intHeight / 2.0, #You can change these but many settings will crash it
	'intCropWidth': int(math.floor(0.98 * intWidth)),
	'intCropHeight': int(math.floor(0.98 * intHeight))
	}

	objectTo = process_autozoom({
	#'dblCenterU': intWidth / 2.0, #Not sure of effect
	#'dblCenterV': intHeight / 2.0, #Not sure of effect
	'dblShift': 100.0, #original
	#'dblShift': 10.0, #try this with large zooms
	#'dblZoom': 40.25, #x50 zoom
	'dblZoom': 1.25, #original
	'objectFrom': objectFrom
	})

	numpyResult = process_kenburns({
	#'dblSteps': numpy.linspace(0.0, 40.0, 800).tolist(), #example very large zoom, lot sof extra frames
	'dblSteps': numpy.linspace(0.0, 1.0, 75).tolist(), #original settings
	#'dblSteps': numpy.linspace(0.0, 20.0, 275).tolist(), # Zoom x20 and more frames 
	'objectFrom': objectFrom,
	'objectTo': objectTo,
	'boolInpaint': True
	})

	moviepy.editor.ImageSequenceClip(sequence=[ numpyFrame[:, :, ::-1] for numpyFrame in numpyResult + list(reversed(numpyResult))[1:] ], fps=float(arguments_strFps)).write_videofile(arguments_strOut, bitrate=arguments_strBitRate)
	# end
