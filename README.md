
---

# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.

[//]: # (Image References)

[image0]: ./output_images/pipeline/Cal_Original "Cal_Original"
[image00]: ./output_images/pipeline/Cal_Undistorted "Cal_Undistorted"
[image1]: ./output_images/pipeline/Undistorted "Undistorted"
[image2]: ./output_images/pipeline/Warped "Warped"
[image3]: ./output_images/pipeline/Combined "Combined"
[image4]: ./output_images/pipeline/lines_win "windows"
[image5]: ./output_images/pipeline/lines_win_fit.jpg "windows fit"
[image6]: ./output_images/pipeline/Original "Original"
[image7]: ./output_images/test/test4.jpg "Original"


---

### Camera Calibration

Using `cv2.findChessboardCorners`, `cv2.calibrateCamera`, `cv2.undistort` I calibrated the camera to obtain undistorted images.

![alt text][image0] ![alt text][image00]

The next image shows an example of undistorted image and the area of interest.

![alt text][image1]

### Pipeline (single images)

#### 1. Perspective transform

After the calibration step, I applied a perspective transform using
`cv2.getPerspectiveTransform` and `cv2.warpPerspective` with source and destination points:

```python
 src = np.array([ [550, 450], [750,450], [1200, 700], [100, 700] ], dtype='float32')
dst = np.array([ [ofs, ofs], [m, ofs], [m, n], [ofs, n] ], dtype='float32')
```
The perspective transform return an image where we can easily measure distances.

![alt text][image2]

#### 2. Gradient and Color thresholding

I used a combination of color and gradient thresholds to generate a binary image. In particular I decided to combine `cv2.Sobel` in the x direction and `cv2.cvtColor` to extract the S-Channel in the HLS color space and the L-Channel in the LAB color space. 

![alt text][image3]

#### 3. Lines detection

Then, using histogram, windows polynomial fit I obtained the two lines of interest.

![alt text][image4]

![alt text][image5]

#### 4. Radius of curvature and position.

In the end I computed radius (in meters) and position of the vehicle respect to the center.

#### 5. Result

Here an example of what I obtained drawing this line on the original image.

![alt text][image7]

---

### Pipeline (video)

Finally I defined a class `Line` where I compute line, radius of curvature and position respect center for every frame; I decided to average over the last 10 frames to obtain a smoother result; I also check for undetected lines in particular frames. 

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### problems and improvement

My implementation is not sophisticated: there is no outliers' detection or some particular check for maximal difference between lines in sequential frames.   
