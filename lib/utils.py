import cv2
import numpy as np

def calibration(n=6, m=9):
    '''camera calibration'''
    calibration_images = glob.glob("./camera_cal/*.jpg")
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image space
    #fig, axs = plt.subplots(5,4, figsize=(16, 11))
    #fig.subplots_adjust(hspace = .2, wspace=.001)
    #axs = axs.ravel()
    objp = np.zeros((n*m, 3), np.float32)
    objp[:,:2] = np.mgrid[0:m, 0:n].T.reshape(-1,2)
    #i = 0
    for cal_img in calibration_images:
        #i = i + 1
        img = mpimg.imread(cal_img)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (m,n), None)
        #img2 = cv2.drawChessboardCorners(img, (m,n), corners, ret)
        #axs[i].imshow(img2)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
    
    img_size = (gray.shape[1], gray.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

def undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def perspective_matrices(img, src=''):
    n, m = img.shape[:-1]
    # top-left top-right bottom-right bottom-left
    if src is '':
        src = np.array([ [550, 450], [750,450], [1200, 700], [100, 700] ], dtype='float32')
    ofs = 0 # offbset for dst points
    dst = np.array([ [ofs, ofs], [m, ofs], [m, n], [ofs, n] ], dtype='float32')
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def perspective(img, M):
    n, m = img.shape[:-1]
    warped = cv2.warpPerspective(img, M, (m,n), flags=cv2.INTER_LINEAR)
    return warped

###################################################################################

def cast_unit8(img):
    m=img.max()
    dims = img.shape
    if len(dims) == 2:
        t=type(img[0][0])
    else:
        t=type(img[0][0][0])
    
    if t==np.uint8 and m > 1:
        return img
    else:
        img = np.uint8( 255 * img/np.max(img) )
    return img

def region(img, vertices):
    mask = np.zeros_like(img)   
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
