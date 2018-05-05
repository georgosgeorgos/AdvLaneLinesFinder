import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def plot(img, converted, title1='Original', title2='Converted', cmap="gray", flag1=False, flag2=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(img)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(converted, cmap=cmap)
    ax2.set_title(title2, fontsize=30)
    
    if flag1 == True:
        mpimg.imsave("output_images/" + title1, img, format="jpg")
    if flag2 == True:
        mpimg.imsave("output_images/" + title2, converted, format="jpg")
        
def cal_example():
    img = mpimg.imread("./camera_cal/calibration2.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    (n, m) = (6,9)
    ret, corners = cv2.findChessboardCorners(gray, (m,n), None)
    img2 = cv2.drawChessboardCorners(img, (m,n), corners, ret)
    dst = undistort(img, mtx, dist)
    return img2, dst

def draw(img, xl, yl, xr, yr, rl, rr, position, Minv, res_old=None, flag=True):
    new_img = np.copy(img)
    try:

        color_warp = np.zeros_like(new_img).astype(np.uint8)
        h, w = new_img.shape[:2]

        if xl != []:
            pts_left = np.array([np.transpose(np.vstack([xl, yl]))])
            cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,0,255), thickness=15)

        if xr != []:
            pts_right = np.array([np.flipud(np.transpose(np.vstack([xr, yr])))])
            cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255,0,0), thickness=15)

        if xl != [] and xr != []:
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))

        if (rl == [] or rr == []): 
            radius = 0 
        else: 
            radius = (rl + rr) / 2
                
        if flag == True:
            font = cv2.FONT_HERSHEY_DUPLEX
            #text = 'radius curvature: ' + '{:04.1f}'.format(radius) + 'm'
            #cv2.putText(new_img, text, (40,70), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
            text = 'distance center: ' + '{:01.2f}'.format(position) + 'm'  
            cv2.putText(new_img, text, (40,120), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

        new_img = new_img.astype('float32')
        newwarp = newwarp.astype('float32')

        result = cv2.addWeighted(new_img, 1, newwarp, 0.4, 0)
        result = np.uint8(result)
    except:
        return res_old
    return result