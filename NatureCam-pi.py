#!/usr/bin/env python
import cv2
import json
import time
import datetime
import os
import numpy as np

# pi specific imports
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
# end of imports

# load configuration file
os.chdir("/home/pi/NatureCam")
config = json.load(open("config.json"))

cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, 1)

# camera
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
time.sleep(config["camera_warmup"])

# buttons
btn1 = 17
btn2 = 22
btn3 = 23
btn4 = 27
btnShutter = btn1
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(btn1, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(btn2, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(btn3, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(btn4, GPIO.IN, GPIO.PUD_UP)

isPi = False

avg = None

mode = 0

lastPhotoTime = 0

minWidth = config["min_width"]
maxWidth = config["max_width"]
minHeight = config["min_height"]
maxHeight = config["max_height"]

activeColour = (255,255,0)
inactiveColour = (100,100,100)
isMinActive = False

def takePhoto(image):
    timestamp = datetime.datetime.now()
    filename = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    filename = filename + ".jpg"
    cv2.imwrite(filename, image)

def detectChangeContours(img):
    global avg
    global lastPhotoTime

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if avg is None:
        avg = gray.copy().astype("float")
        # remember to truncate capture for Pi
        return img
    
    # add to accumulation model and find the change
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold, dilate and find contours
    thresh = cv2.threshold(frameDelta, config["delta_threshold"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find largest contour
    largestContour = getLargestContour(cnts)

    if largestContour is None:
        return img

    (x, y, w, h) = cv2.boundingRect(largestContour)
        
    # if the contour is too small, just return the image.
    if w > maxWidth or w < minWidth or h > maxHeight or h < minHeight:
        return img

    # otherwise, draw the rectangle
    if time.time() - lastPhotoTime > config['min_photo_interval_s']:
        takePhoto(img)
        lastPhotoTime = time.time()

    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

    return img

def getLargestContour(contours):
    if not contours:
        return None
    else:
        areas = [cv2.contourArea(c) for c in contours]
        maxIndex = np.argmax(areas)
        return contours[maxIndex]
        
def displayMinMax(img):
    if isMinActive is True:
        minColour = activeColour 
        maxColour = inactiveColour
    else:
        minColour = inactiveColour
        maxColour = activeColour
    
    cv2.rectangle(img, (320/2-minWidth/2,240/2-minHeight/2), (320/2+minWidth/2,240/2+minHeight/2), minColour, 2)
    cv2.rectangle(img, (320/2-maxWidth/2,240/2-maxHeight/2), (320/2+maxWidth/2,240/2+maxHeight/2), maxColour, 2)
    return img

def increaseMinMax(increment):
    global minWidth
    global minHeight
    global maxWidth
    global maxHeight
    
    if isMinActive is True:
        minWidth = minWidth + increment
        minHeight = minHeight + increment
        if minWidth > maxWidth:
            minWidth = maxWidth
            minHeight = maxHeight
    else:
        maxWidth = maxWidth + increment
        maxHeight = maxHeight + increment
        if maxWidth > 320:
            maxWidth = 320
            maxHeight = 320
        if maxHeight >= 240:
            maxHeight = 240

def decreaseMinMax(increment):
    global minWidth
    global minHeight
    global maxWidth
    global maxHeight
    
    if isMinActive is True:
        minWidth = minWidth - increment
        minHeight = minHeight - increment
        if minWidth < 0:
            minWidth = 0
            minHeight = 0
    else:
        maxWidth = maxWidth - increment
        maxHeight = maxHeight - increment
        if maxWidth < minWidth:
            maxWidth = minWidth
            maxHeight = minHeight
        if maxWidth < 240:
            maxHeight = maxWidth
        elif maxWidth >= 240:
            maxHeight = 240

# main loop
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # get new frame
    image = frame.array
    # end of new frame

    if mode == 0:
        image = displayMinMax(image)
    # armed mode
    if mode == 1:
        image = detectChangeContours(image)

    cv2.imshow("Output", image)

    if GPIO.input(btnShutter) == False:
        mode = not mode
        time.sleep(0.5)
    if GPIO.input(btn2) == False:
        isMinActive = not isMinActive
        time.sleep(0.25)
    if GPIO.input(btn3) == False:
        decreaseMinMax(5)
    if GPIO.input(btn4) == False:
        increaseMinMax(5)

    # clear buffer
    rawCapture.truncate(0)
    key = cv2.waitKey(10)
    # end of loop

# cleanup
cv2.destroyWindow("Output")
