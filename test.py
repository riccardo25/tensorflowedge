
import numpy as np
import tensorflow as tf
import math
import cv2
import os
import glob

path = "image.jpg"
img_size = 500


def show(img, name = "image", wait = True):
    cv2.imshow(name,img)
    
    if(wait):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def average(img):
    np.power(img, 2)
    sum = np.sum(img)
    return sum

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
#image = cv2.resize(image, (img_size, img_size),0,0, cv2.INTER_LINEAR)
image = image.astype(np.float32)
image = np.multiply(image, 1.0/255.0) 
show(image, "original", False)
print(image.shape)

tf.reset_default_graph()

#filtro di sobel
# Write the kernel weights as a 2D array. 
kernel_h = np.array([3, 3])
kernel_h = [ [1,2,1], [0,0,0], [-1,-2,-1] ]

kernel_v = np.array([3, 3])
kernel_v = [ [1,0,1], [2,0,-2], [-1,0,-1] ]

#fitro di roberts
#kernel_h = np.array([2, 2])
#kernel_h = [ [1,0], [0,-1] ]

#kernel_v = np.array([2, 2])
#kernel_v = [ [1,0], [0,-1] ]

# Kernel weights

if len(kernel_h) == 0 or len(kernel_v) == 0:
    print('Please specify the kernel!')

input_placeholder = tf.placeholder( dtype=tf.float32, shape=(1, image.shape[0], image.shape[1], 1))

with tf.name_scope('convolution'):
    conv_w_h = tf.constant(kernel_h, dtype=tf.float32, shape=(len(kernel_h), len(kernel_h), 1, 1))
    conv_w_v = tf.constant(kernel_v, dtype=tf.float32, shape=(len(kernel_v), len(kernel_v), 1, 1))    
    output_h = tf.nn.conv2d(input=input_placeholder, filter=conv_w_h, strides=[1, 1, 1, 1], padding='SAME')
    output_v = tf.nn.conv2d(input=input_placeholder, filter=conv_w_v, strides=[1, 1, 1, 1], padding='SAME')
    
    #output_h = tf.nn.max_pool(value=output_h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #output_v = tf.nn.max_pool(value=output_v, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    ampli = 1
    bias = 0

    output_h = tf.nn.relu(output_h*ampli-bias)
    output_v = tf.nn.relu(output_v*output_v*ampli-bias)

with tf.Session() as sess:
    result_h = sess.run(output_h, feed_dict={ input_placeholder: image[np.newaxis, :, :, np.newaxis]})
    result_v = sess.run(output_v, feed_dict={ input_placeholder: image[np.newaxis, :, :, np.newaxis]})

show(result_v[0, :, :, 0], "vertical", False)

result_lenght = ((result_v**2) + (result_h**2))**0.5  

result_lenght[0, 0, :, 0] = 0
result_lenght[0, result_lenght.shape[1]-1, :, 0] = 0
result_lenght[0, :, 0, 0] = 0
result_lenght[0, :, result_lenght.shape[2]-1,  0] = 0


show(result_lenght[0, :, :, 0], "lenght", True)
#result_angle = (np.arctan(result_v/(result_h+0.00000001)))
#show(result_angle[0, :, :, 0], "angle", True)

