import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def hsvscale(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv

def hlsscale(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls

def labscale(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return lab

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def sobel_thresh(img, sobel_kernel=5, flag="x", thresh_min=0, thresh_max=255):
    gray = grayscale(img)    
    if flag == "x":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel = np.absolute(sobelx)
    elif flag == "y":
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel = np.absolute(sobely)
    ## magnitude
    elif flag == "m":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel = np.sqrt(sobelx**2 + sobely**2)
    ## direction
    elif flag == "d":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        direction = np.arctan2(abs_sobely, abs_sobelx)
    
    if flag != "d":
        scaled_sobel = np.uint8(255*sobel / np.max(sobel))
    else:
        scaled_sobel = direction
        
    mask = np.zeros_like(scaled_sobel)
    mask[ (scaled_sobel > thresh_min) & (scaled_sobel <= thresh_max) ] = 1
    return mask

def color_thresh(img, flag="s", thresh_min=0, thresh_max=255):
    hls = hlsscale(img)
    lab = labscale(img)
    if flag == "s":
        channel = hls[:,:,2]
    elif flag == "l":
        channel = lab[:,:,0]
    elif flag == "g":
        channel = grayscale(img)
        
    ### attention format image...different behaviour (png, jpg, ecc)
    scaled_channel = channel.copy()
    mask = np.zeros_like(scaled_channel)
    mask[(scaled_channel > thresh_min) & (scaled_channel <= thresh_max)] = 1
    return mask

def f_thresh(gauss, x, m, g, l, s, flag_video=None):
    thresh_x = sobel_thresh(gauss, flag="x", thresh_min=x, thresh_max=255)
    thresh_m = sobel_thresh(gauss, flag="m", thresh_min=m, thresh_max=255)
    
    gray = color_thresh(gauss, "g", thresh_min=g, thresh_max=255)
    color_l = color_thresh(gauss, "l", thresh_min=l, thresh_max=255)
    # for yellow lines
    color_s = color_thresh(gauss, "s", thresh_min=s, thresh_max=100)
    
    combined = np.zeros_like(thresh_x)
    ## decide if (and/or) between gradient and color thresholds
    #if flag_video == 'kvideo1':
    #    color_s[:, 640:] = 0
    #    combined[((color_l == 1) | (gray == 1) | (color_s == 1))] = 1
    #else:
    combined[ ( (thresh_x == 1) | (thresh_m == 1) ) | ( (color_l == 1) | (gray == 1) | (color_s == 1) ) ] = 1
    return combined