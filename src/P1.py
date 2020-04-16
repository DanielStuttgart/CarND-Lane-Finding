#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[9]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images  
# `cv2.cvtColor()` to grayscale or change color  
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[73]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[74]:


import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[102]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

## Calculate linear equation for hough-line with parameter slope and offset
# y = mx + b = slope * x + offset
# 1) slope = (y2 - y1) / (x2 - x1)
# 2) offset = y - slope * x
# furthermore: slope > 0.5, since road is nearly straight ahead and dy > dx
def get_linear_eq(line, min_slope = 0.3):
    #print(line[0])
    if line[0][2] - line[0][0] != 0:
        slope = (line[0][3]-line[0][1])/(line[0][2] - line[0][0])
        if np.abs(slope) < min_slope:
            slope = np.nan
    else:
        slope = np.nan
    offset = line[0][1] - (slope * line[0][0])
    return slope, offset

def find_lanes(image, debug_info, use_kmeans = 1):
    # debug info --> show all images 
    #debug_info = True

    # show original image        
    if debug_info:
        print('Original image')
        # show original image
        plt.figure()
        plt.imshow(image)
        plt.show(block=False)


    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]/3, imshape[0]/16*10), (imshape[1]/3*2, imshape[0]/16*10), (imshape[1],imshape[0])]], dtype=np.int32)
    # 3) look for intersection at y = ims/8*5 (top) and y = imheight
    y_intersect_top = vertices[0][1,1]
    y_intersect_bottom = vertices[0][0,1]
    
    # test roi with original image
    img_roi = region_of_interest(image, vertices)
    if debug_info:
        print('ROI-Filter')
        plt.figure()
        plt.imshow(img_roi)
        plt.show(block=False)

    # convert to grayscale
    img_gray = grayscale(image)
    if debug_info:
        print('Gray-scale image')
        plt.figure()
        plt.imshow(img_gray,cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)

    # apply blur filter
    kernel_size = 5
    img_blur = gaussian_blur(img_gray, kernel_size)
    if debug_info:
        print('Gaussian blur')
        plt.figure()
        plt.imshow(img_blur,cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)

    # apply canny edge detection
    low_threshold = 150
    high_threshold = 200
    img_canny = canny(img_blur, low_threshold, high_threshold)
    if debug_info:
        print('Canny Edges')
        plt.figure()
        plt.imshow(img_canny,cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)

    # apply region of interest
    img_roi = region_of_interest(img_canny, vertices)
    if debug_info:
        print('ROI-Filte')
        plt.figure()
        plt.imshow(img_roi,cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)

    # apply hough transform
    rho = 1. # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 6 # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 12 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    #line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # calculate lines for extrapolation
    #lines_hough = cv2.HoughLinesP(img_roi, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)    
    #print(lines_hough)
    
    img_hough, lines_hough = hough_lines(img_roi, rho, theta, threshold, min_line_len, max_line_gap)
    
    # calculate linear equation for slope and offset
    linear_eq = np.array([])
    for line in lines_hough:
        linear_eq = np.append(linear_eq, get_linear_eq(line))
    
    # remove nan
    linear_eq = linear_eq[~np.isnan(linear_eq)]            

    # reshape for new order [slope, offset]
    linear_eq = np.reshape(linear_eq, (-1,2))
    #print(linear_eq[:,0])
    if debug_info:
        plt.figure()        
        plt.plot(linear_eq[:,0], linear_eq[:,1], 'o')     

    ###################
    # New version: don't rely on positive / negative slope but find 2x clusters for left and right lane    
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=500, n_init=20, random_state=None) 
    pred_lanes = kmeans.fit_predict(linear_eq)   
    
    if debug_info:
        # plot result from kmeans --> [:,0]: slope; [:,1]: offset
        plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], '*r')    

    ###################
    # Outlier-Detection based on first kmeans-clusters      
    #offset_q1, offset_q3 = np.quantile(linear_eq[:,1], [0.25,0.75]) 
    # iterate among dimensions of linear_eq
    for i in range(2):
        slope_q1, slope_q3 = np.quantile(linear_eq[kmeans.labels_==i,0], [0.25,0.75])      
        offset_q1, offset_q3 = np.quantile(linear_eq[kmeans.labels_==i,1], [0.25,0.75])            
        slope_iqr = slope_q3 - slope_q1
        offset_iqr = offset_q3 - offset_q1

        # calculate boundaries for outlier-detection
        boundary_factor = 1.  # 1. or 1.5
        slope_low = slope_q1 - (boundary_factor * slope_iqr)            # lower_bound = q1 -(1.5 * iqr) 
        slope_high = slope_q3 + (boundary_factor * slope_iqr)           # upper_bound = q3 +(1.5 * iqr)
        offset_low = offset_q1 - (boundary_factor * offset_iqr)
        offset_high = offset_q3 + (boundary_factor * offset_iqr)

        # filter slopes and offset in linear_eq in order to be within boundaries
        filtered_slopes = np.logical_and(linear_eq[kmeans.labels_==i,0] < slope_high, linear_eq[kmeans.labels_==i,0] > slope_low)
        filtered_offset = np.logical_and(linear_eq[kmeans.labels_==i,1] < offset_high, linear_eq[kmeans.labels_==i,1] > offset_low)
        # calc median of cluster without outlier (detected by filtered_slopes_1)
        # if no offset / slope relieves after filtering, consider original clusters
        if sum(filtered_slopes > 0):
            kmeans.cluster_centers_[i,0] = np.median(linear_eq[kmeans.labels_==i,0][filtered_slopes])
        # check value boundaries [-5 < slope < 5]
        elif np.abs(kmeans.cluster_centers_[i,0]) > 5:
            kmeans.cluster_centers_[i,0] = np.sign(kmeans.cluster_centers_[i,0]) * 5        
        
        if sum(filtered_offset > 0):
            kmeans.cluster_centers_[i,1] = np.median(linear_eq[kmeans.labels_==i,1][filtered_offset])   
        # check value boundaries [-1000 < offset < 1000]
        elif np.abs(kmeans.cluster_centers_[i,1]) > 1000:
            kmeans.cluster_centers_[i,1] = np.sign(kmeans.cluster_centers_[i,1]) * 1000

        # check on zero for slopes (would raise division-by-zero)
        if kmeans.cluster_centers_[i,0] == 0:
            kmeans.cluster_centers_[i,0] = 0.01

    if debug_info:
        plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], '*c')
        plt.show(block=False)   

    # find values in first cluster
    #linear_eq[np.logical_and(kmeans.labels_==0, kmeans.labels_==1),0]
     #   filtered_points = np.logical_and(filtered_slopes, filtered_offset)
      ##  filtered_linear_eq = linear_eq[filtered_points,:]


    # calculate lines to visualize at y-coord of ROI 
    # 4) x_intersect_low = (y_intersect - b) / slope_
    if use_kmeans == False:
        if med_slope_neg != np.nan and med_slope_neg != np.nan and med_offset_neg != np.nan and med_offset_pos != np.nan:
            x_intersect_neg_bottom = (y_intersect_bottom - med_offset_neg) / med_slope_neg
            x_intersect_pos_bottom = (y_intersect_bottom - med_offset_pos) / med_slope_pos
            x_intersect_neg_top = (y_intersect_top - med_offset_neg) / med_slope_neg
            x_intersect_pos_top = (y_intersect_top - med_offset_pos) / med_slope_pos
        else:
            x_intersect_neg_bottom = 0
            x_intersect_pos_bottom = 0
            x_intersect_neg_top = 0
            x_intersect_pos_top = 0
    else:
        x_intersect_neg_bottom = (y_intersect_bottom - kmeans.cluster_centers_[0,1]) / kmeans.cluster_centers_[0,0]
        x_intersect_pos_bottom = (y_intersect_bottom - kmeans.cluster_centers_[1,1]) / kmeans.cluster_centers_[1,0]
        x_intersect_neg_top = (y_intersect_top - kmeans.cluster_centers_[0,1]) / kmeans.cluster_centers_[0,0]
        x_intersect_pos_top = (y_intersect_top - kmeans.cluster_centers_[1,1]) / kmeans.cluster_centers_[1,0]

    extrap_lanes = np.array([[[x_intersect_neg_bottom, y_intersect_bottom, x_intersect_neg_top, y_intersect_top]], 
        [[x_intersect_pos_bottom, y_intersect_bottom, x_intersect_pos_top, y_intersect_top]]], dtype=np.int32)

    print('Hough Transform')
    if debug_info:
        plt.figure()
        plt.imshow(img_hough,cmap='gray', vmin=0, vmax=255)            
        #draw_lines(img_hough, extrap_lanes, color=[0, 255, 0], thickness=2)
        #plt.imshow(img_hough)
        plt.show(block=False)

    if debug_info:
        print('Hough Transform')
        plt.figure()
        plt.imshow(img_hough,cmap='gray', vmin=0, vmax=255)
        #plt.Line2D()
        plt.show(block=False)

    # show final image
    img_weight = weighted_img(img_hough, image)
    if debug_info:
        print('Final image')
        plt.figure()
        plt.imshow(img_weight)
        plt.show(block=False)
        
    return img_weight, extrap_lanes

# start interactive mode for VS Code
#plt.ion()

img_names = os.listdir("test_images/")

for name in img_names:
    print('test_images/' + name)
    img = mpimg.imread('test_images/' + name)
    new_img, extrap_lanes = find_lanes(img, True, True)
    draw_lines(new_img, extrap_lanes, color=[0, 255, 0], thickness=2)

    plt.figure()
    plt.imshow(img)
    plt.show(block=False)    
    plt.figure()
    plt.imshow(new_img)
    plt.show()


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[64]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[103]:


def process_image(image, improved = True):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    cv2.normalize(image,  image, 0, 255, cv2.NORM_MINMAX)
    result, extrap_lanes = find_lanes(image, False, True)
    if improved == True:
        result = image
        draw_lines(result, extrap_lanes, color=[0, 255, 0], thickness=2)

    return result


# Let's try the one with the solid white lane on the right first ...

# In[104]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[105]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[106]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[107]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[108]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[109]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[ ]:




