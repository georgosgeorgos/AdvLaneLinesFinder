import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import HTML
from moviepy.editor import VideoFileClip

from lib import thresh, utils, vis
from lib import KFilter as kf

def select_pixel_center(hist):
    '''choose the initial center considering the integral of the histogram and not only the picks'''
    indices = np.where(hist > 50)[0]
    d = []
    list_index = [indices[0]]

    for i in range(1, len(indices)):
        new = indices[i]
        old = indices[i - 1]
        if new == (old + 1):
            list_index.append(indices[i])
        else:
            value = list_index[ len(list_index) // 2 ]
            area = sum(hist[list_index])
            d.append((area, value))
            list_index = [indices[i]]

    value = list_index[ len(list_index) // 2 ]
    area = sum(hist[list_index])
    d.append((area, value))
    return max(d)[1]


def routine_starting_centers(h, c, c_min, c_max, flag, ofs):
    if len(h) != 0:
        if flag == False:
            center = h.argmax() + ofs
        else:
            try:
                center = select_pixel_center(h) + ofs
            except IndexError:
                center = h.argmax() + ofs
    # interval for center
    if (center < c_min or center > c_max):
        center = c
    return center
    

def starting_centers(image, cl=200, cr=1000, flag=False, frame=None):
    '''choose starting pixel for lines using picks histograms'''
    n, m = image.shape[:2]
    ofs = m // 2
    
    left_center = None
    right_center = None
    
    if frame == 1:
        hist = np.sum(image[n//4:], axis=0)
    else:
        hist = np.sum(image[n//2:], axis=0)
    t=0 #hist.shape[0]//2
    
    left_hist = hist[:(ofs)]
    left_center = routine_starting_centers(left_hist, cl, 160, 400, flag, 0)   # 160-360
    
    right_hist = hist[(ofs):]
    right_center = routine_starting_centers(right_hist, cr, 700, 1200, flag, ofs) # 900-1200
    
    return left_center, right_center

def window_routine(gauss, image, center, height_win, t, k=100, flag=False):
    '''generate a new window box given t and a center'''
    left_limit = (- k + center)
    right_limit = (center + k)
    
    if left_limit < 50: left_limit = 50
    if right_limit > 1230: right_limit = 1230
    
    bottom_left  = (left_limit, (height_win * t))
    bottom_right = (right_limit, (height_win * t))
    upper_left   = (left_limit, (height_win) * (t + 1))
    upper_right  = (right_limit, (height_win) * (t + 1))
    
    if flag == True:
        cv2.rectangle(gauss,bottom_left,upper_right,(0,0,255), 6)
    
    win = image[ (height_win * t):((height_win) * (t + 1)), left_limit:right_limit ]
    return win

def window_extract(fltr, gauss, win, line, old_center, height_win, t, k=80, minpix=50, flag=False): # k=100
    '''extract non-zero pixels (change function name)'''
    # respect window
    line_w = win.nonzero()
    hist = np.sum(win, axis=0)
    
    if len(hist) == 0:
        return old_center, line
    
    # relative to the window
    center = hist.argmax()
        
    if len(line_w[0]) > minpix:
        # absolute to the image
        new_center = (- k + old_center) + center
        
        if new_center < 50:
            new_center = old_center
        if new_center > 1230:
            new_center = old_center

        ## check maximum gap between window centers
        delta = new_center - old_center
        if np.abs(delta) > (1 / 2) * k:
            # smooth strong variations
            new_center = int( old_center + 0.3 * delta )  # 0.1 


        x = line_w[1] + ( - k + old_center)
        y = line_w[0] + (height_win) * (t)
        
        # filter
        #if x != [] and y != []:
        #    fx, xx = np.histogram(x)
        #    xx = int(xx[np.argmax(fx)])
        #    yy = int((height_win) * (t + 0.5)) 
        #    if fltr.likelihood(xx) > 0.04:
        #        fltr.run(xx)
        #        xx = fltr.position()
        #    
        #    line[0].extend([xx])
        #    line[1].extend([yy])
        
        line[0].extend(x)
        line[1].extend(y)
    else:
        new_center = old_center
        
    
    
    if flag == True:
        # draw line on window center
        cv2.line(gauss, (new_center, (height_win)*(t)), (new_center, (height_win)*(t+1)), (255,0,0), 6)
    return new_center, line

def detect_lines(fltr_left, fltr_right, gauss, image, left_center_old, right_center_old, 
                 n_windows=10, k=80):
    n, m = image.shape[:2]
    height_win = n // n_windows
    
    line_left, line_right  = [ [], [] ], [ [], [] ]
    radius_left, radius_right  = [], []
    xl, yl, xr, yr = [], [], [], []
    
    # determine starting pixel
    left_center, right_center = starting_centers(image, left_center_old, right_center_old)
    
    # handle corner cases
    if left_center == None and right_center == None:
        xl, yl, xr, yr, radius_left, radius_right, left_center, right_center
        
    if left_center == None:
        left_center = left_center_old
    left_center_start = left_center
        
    if right_center == None:
        right_center = right_center_old
    right_center_start = right_center
    
    # for every window (starting from the bottom)
    for t in range((n_windows-1), -1, -1):
        # create window
        win_left = window_routine(gauss, image, left_center, height_win, t, flag=True)
        win_right = window_routine(gauss, image, right_center, height_win, t, flag=True)
        # select pixels for line
        left_center_new, line_left = window_extract(fltr_left, gauss, win_left, line_left, 
                                                     left_center, height_win, t, flag=True)
        right_center_new, line_right = window_extract(fltr_right, gauss, win_right, line_right,
                                                       right_center, height_win, t, flag=True)
        
        delta = right_center_new - left_center_new
        delta_left = left_center_new - left_center
        delta_right = right_center_new - right_center
        
        # max distance lines  add max delta for consecutive windows
        if (delta > 700 and delta < 1000):  # 800 900
            if np.abs(delta_left) < (1 / 2) * k:
                left_center = int(left_center_new)
            else:
                left_center = int(left_center + 0.3 * delta_left)  # 0.1
                
            if np.abs(delta_right) < (1 / 2) * k:
                right_center = int(right_center_new)
            else:
                right_center = int(right_center + 0.3 * delta_right) # 0.1

    left_fit, left_fit_radius = polynomial_fit(line_left)
    right_fit, right_fit_radius = polynomial_fit(line_right)
    
    return left_fit, right_fit, left_fit_radius, right_fit_radius, left_center_start , right_center_start 

def polynomial_fit(line, minpix=100):
    # for polyfit...I want a polynomial where the indipendent variable is x
    ym_per_pix = 30 / 720 # meters per pixel
    xm_per_pix = 3.7 / 700
    
    x = np.array(line[0], dtype="int32")
    y = np.array(line[1], dtype="int32")
    # if the detected line has more than minpix
    if len(x) > minpix:
        # fit line coeff
        fit = np.polyfit(y, x, 2)
        # fit radius coeff
        fit_radius = np.polyfit( (y * ym_per_pix), (x * xm_per_pix), 2)
    else:
        fit, fit_radius = [], []
    return fit, fit_radius

def generate_line(fit, n=720):
    if fit == []:
        return [], []
    y = np.linspace(0, n)
    x = fit[2] + fit[1] * y + fit[0] * y**2
    y = y.astype("int32")
    x = x.astype("int32")
    return x, y

def generate_radius(fit_radius, n=720):
    ym_per_pix = 30 / 720 # meters per pixel
    xm_per_pix = 3.7 / 700
    if fit_radius == []:
        return []
    y = np.linspace(0, n)
    radius = (1 + (2 * fit_radius[0] * (y * ym_per_pix) + fit_radius[1])**2) * (3/2) / (2 * fit_radius[0])
    return radius

def routine_g(left_fit, right_fit, left_fit_radius, right_fit_radius, 
              left_center_start, right_center_start):
    ofs = 1280 // 2
    xm_per_pix = 3.7 / 700
    xl, yl = generate_line(left_fit)
    xr, yr = generate_line(right_fit)

    radius_left = generate_radius(left_fit_radius)
    radius_right = generate_radius(right_fit_radius)
    
    if radius_left == []: rl = []
    else: rl = radius_left[len(radius_left)//2]
    
    if radius_right == []: rr = []
    else: rr = radius_right[len(radius_right)//2]

    position = (left_center_start + right_center_start) / 2 - ofs
    position_m = position * xm_per_pix
    return xl, yl, xr, yr, rl, rr, position_m

def pipeline(dst, src, x, m, g, l, s,
             cl_old=200, cr_old=1000, flag=False, flag2=None):
    # perspective matrices
    M, Minv = utils.perspective_matrices(dst, src)
    # perspective transform
    warped = utils.perspective(dst, M)
    # smooth
    gauss = thresh.gaussian_blur(warped, 5)
    # gradient and color threshold
    combined = thresh.f_thresh(gauss, x, m, g, l, s, flag2)
    
    # filters
    fl = kf.Filter(x0=cl_old)
    fr = kf.Filter(x0=cr_old)
    # compute coeff and center windows
    res = detect_lines(fl, fr, gauss, combined, cl_old, cr_old)
    left_fit, right_fit, left_fit_r, right_fit_r, cl_start, cr_start = res
    
    xl, yl, xr, yr, rl, rr, p = routine_g(left_fit, right_fit, left_fit_r, right_fit_r, cl_start, cr_start)
    if flag == True:
        # for frames
        return gauss, xl, yl, xr, yr, rl, rr, cl_start, cr_start, p, Minv
    else:
        # for videos
        return xl, yl, xr, yr, rl, rr, cl_start, cr_start, p, Minv
    
def pipeline_first_frame(dst, src, x, m, g, l, s):
    '''scan the first frame to find the left and right centers'''
    # perspective matrices
    M, Minv = utils.perspective_matrices(dst, src)
    # perspective transform
    warped = utils.perspective(dst, M)
    # smooth
    gauss = thresh.gaussian_blur(warped, 5)
    # gradient and color threshold
    combined = thresh.f_thresh(gauss, x, m, g, l, s)
    
    left_center, right_center = starting_centers(combined)
    return left_center, right_center

def first_centers(video_input, src, mtx, dist, x, m, g, l, s):
    first_frame = video_input.get_frame(0)
    dst = utils.undistort(first_frame, mtx, dist)
    cl_start, cr_start = pipeline_first_frame(dst, src, x, m, g, l, s)
    return cl_start, cr_start

def load_parameters():
    calibration = pickle.load(open( "calibration.p", "rb" ))
    mtx = calibration["mtx"]
    dist = calibration["dist"]
    return mtx, dist

def main_preprocess(video, src, x, m, g, l, s):
    mtx, dist = load_parameters()
    cl_start, cr_start = first_centers(video, src, mtx, dist, x, m, g, l, s)
    return cl_start, cr_start, mtx, dist