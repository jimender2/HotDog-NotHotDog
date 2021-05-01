import cv2 as cv
import numpy
import glob
import os

imageList = glob.glob("Dataset/**/*.jpg", recursive = True)

for i in imageList:
    img = cv.imread(i)
    norm = numpy.zeros((800,800))

    # normalize image
    final = cv.normalize(img, norm, 0, 255, cv.NORM_MINMAX)

    width = 100
    height = 100
    dim = (width, height)

    # resize image
    resized = cv.resize(final, dim, interpolation = cv.INTER_AREA)

    # convert to grayscale
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    # write to disk
    cv.imwrite(i.replace("Dataset", "Normalized Dataset"), gray)