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

arguments_strFps = '24'
arguments_strInDir = './images/'
arguments_strOutDir = './videos/'
arguments_strZipName = 'download.zip'
arguments_strNoZip = 'N'
files_to_zip = []

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--fps' and strArgument != '': arguments_strFps = strArgument # allow FPS value selection from cmd line
	if strOption == '--indir' and strArgument != '': arguments_strInDir = strArgument # input directory for processing
	if strOption == '--outdir' and strArgument != '': arguments_strOutDir = strArgument # output directory for videos
	if strOption == '--zipname' and strArgument != '': arguments_strZipName = strArgument # name for the zip file
	if strOption == '--nozip' and strArgument != '': arguments_strNoZip = strArgument
	# end

	##########################################################

if __name__ == '__main__':

	# check if the output directory exists if not, create it
	if os.path.exists(arguments_strOutDir) and os.access(arguments_strOutDir, os.R_OK):
		print("Output directory "+arguments_strOutDir+" exists.")
	else:
		print("Output directory does not exist or is not writable.")
		print("Attempting to create it")
		try:
			os.makedirs(arguments_strOutDir)
			print("Directory "+arguments_strOutDir+" created.")
		except:
			print("CREATING DIRECTORY FAILED!")
			exit()

	# loop through the files in the inDir, find .jpg or .png and process them, leaving the output in outDir
	for file in os.listdir(arguments_strInDir):
		inFile = os.fsdecode(file)

		if inFile.endswith(".jpg") or inFile.endswith(".png"):
			numpyImage = cv2.imread(filename=arguments_strInDir+inFile, flags=cv2.IMREAD_COLOR)

			intWidth = numpyImage.shape[1]
			intHeight = numpyImage.shape[0]

			dblRatio = float(intWidth) / float(intHeight)

			intWidth = min(int(1024 * dblRatio), 1024)
			intHeight = min(int(1024 / dblRatio), 1024)

			numpyImage = cv2.resize(src=numpyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

			process_load(numpyImage, {})

			objectFrom = {
			'dblCenterU': intWidth / 2.0,
			'dblCenterV': intHeight / 2.0,
			'intCropWidth': int(math.floor(0.97 * intWidth)),
			'intCropHeight': int(math.floor(0.97 * intHeight))
			}

			objectTo = process_autozoom({
			'dblShift': 100.0,
			'dblZoom': 1.25,
			'objectFrom': objectFrom
			})

			numpyResult = process_kenburns({
			'dblSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
			'objectFrom': objectFrom,
			'objectTo': objectTo,
			'boolInpaint': True
			})

			moviepy.editor.ImageSequenceClip(sequence=[ numpyFrame[:, :, ::-1] for numpyFrame in numpyResult + list(reversed(numpyResult))[1:] ], fps=arguments_strFps).write_videofile(outFile)


	if arguments_strNoZip == 'n' or arguments_strNoZip == 'N':
		# zip the files up for download
		print("Compressing the zip file...")
		for file in os.listdir(arguments_strOutDir):

			file = os.fsdecode(file)

			if file.endswith(".mp4"):
				files_to_zip.append(arguments_strOutDir+file)

				with zipfile.ZipFile(arguments_strZipName, 'w') as zip:

					for file_to_zip in files_to_zip:
						print("Zipping: "+file_to_zip)
						zip.write(file_to_zip)

						print("Files zipped.")
