import cv2
import json
import time
import datetime
import os
import numpy as np

# configuration file
config = json.load(open("config.json"))
# end of configuration file

# window
cv2.namedWindow("Output")
# end of window

# camera
capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)
if capture.isOpened():
    rval, frame = capture.read()
else:
    rval = False
time.sleep(config["camera_warmup"])
# end of camera

isPi = False

avg = None

mode = 0

minWidth = config["min_width"]
maxWidth = config["max_width"]
minHeight = config["min_height"]
maxHeight = config["max_height"]

activeColour = (255,255,0)
inactiveColour = (100,100,100)
isMinActive = False

def detectChangeContours(img):
    global avg

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
while rval:
    # new frame
    rval, image = capture.read()
    # end of new frame

    if mode == 0:
        image = displayMinMax(image)
    # armed mode
    if mode == 1:
        image = detectChangeContours(image)

    cv2.imshow("Output", image)

    # wait for keys
    key = cv2.waitKey(10)
    if key == ord('1'):
        mode = not mode
    if key == ord('2'):
        isMinActive = not isMinActive
    if key == ord('3'):
        decreaseMinMax(5)
    if key == ord('4'):
        increaseMinMax(5)
    if key == 27:
        break
    # end of loop

# cleanup
cv2.destroyWindow("Output")
