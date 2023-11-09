import numpy as np
import os
import matplotlib.image as mpimg

print(os.getcwd())

def processImage(image_path):
    img = img2grayVect(image_path)
    radius = min(img.shape) / 2 - 1
    center = (radius + 1, radius + 1)
    hooks = getHooks(50, radius, center)
    numLines = 100
    current_hook = hooks[0]
    output = []
    for _ in range(numLines):
        next_hook = findBestHook(img, hooks, current_hook)
        output.append((current_hook, next_hook))
        current_hook = next_hook
    return output
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def img2grayVect(image_path):
    img_rgb = mpimg.imread(image_path)
    return rgb2gray(img_rgb)

def getHooks(numHooks, radius, center):
    step = 2 * np.pi / numHooks #Evenly divide a circle
    theta_vector = np.arange(0, 2 * np.pi, step)
    x_vector = radius * np.cos(theta_vector) + center[0]
    y_vector = radius * np.sin(theta_vector) + center[1]
    return np.array((x_vector, y_vector)).T

def findBestHook(imgGreyScalePixels, hooks, current_hook):
    score = float("inf")
    bestHook = current_hook
    for new_hook in hooks:
        if all(new_hook != current_hook):
            pixels, weights = getPixels(current_hook, new_hook)
            line = 255 * weights # greyscale vector we want to minimize
            imgLine = []
            for x, y in pixels:
                imgLine.append(imgGreyScalePixels[x][y])
            imgLine = np.array(imgLine)
            minimizer = imgLine - line
            if (np.linalg.norm(minimizer) < score):
                bestHook = new_hook
                score = np.linalg.norm(minimizer)
    return bestHook


def getPixels(hook1, hook2):
    return xiaoline(hook1[0], hook1[1], hook2[0], hook2[1])

# Python Xiaoline algorithm - https://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
def xiaoline(x0, y0, x1, y1):
        #integer part of x
        def ipart(x):
            return int(x)

        def round(x):
            return ipart(x + 0.5)

        #fractional part of x
        def fpart(x):
            return x - ipart(x)

        def rfpart(x):
            return 1 - fpart(x)
        x=[]
        y=[]
        b=[]
        dx = x1-x0
        dy = y1-y0
        steep = abs(dx) < abs(dy)

        if steep:
            x0,y0 = y0,x0
            x1,y1 = y1,x1
            dy,dx = dx,dy

        if x0 > x1:
            x0,x1 = x1,x0
            y0,y1 = y1,y0

        gradient = float(dy) / float(dx)  # slope

        """ handle first endpoint """
        xend = round(x0)
        yend = y0 + gradient * (xend - x0)
        xgap = rfpart(x0 + 0.5)
        xpxl0 = int(xend)
        ypxl0 = int(yend)
        x.append(xpxl0)
        y.append(ypxl0)
        b.append(rfpart(yend) * xgap)

        x.append(xpxl0)
        y.append(ypxl0+1)
        b.append(fpart(yend) * xgap)
        intery = yend + gradient

        """ handles the second point """
        xend = round (x1);
        yend = y1 + gradient * (xend - x1);
        xgap = fpart(x1 + 0.5)
        xpxl1 = int(xend)
        ypxl1 = int (yend)
        x.append(xpxl1)
        y.append(ypxl1)
        b.append(rfpart(yend) * xgap) 

        x.append(xpxl1)
        y.append(ypxl1 + 1)
        b.append(fpart(yend) * xgap)

        """ main loop """
        for px in range(xpxl0 + 1 , xpxl1):
            x.append(px)
            y.append(int(intery))
            b.append(rfpart(intery))

            x.append(px)
            y.append(int(intery) + 1)
            b.append(fpart(intery))
            
            intery = intery + gradient;

        if steep:
            y,x = x,y
        x = np.array(x)
        y = np.array(y)
        b = np.array(b)
        points = np.array((x,y)).T
        return points, b

print(processImage("Portrait1B&W.jpg"))
    
