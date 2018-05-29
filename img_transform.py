from code import process_img
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

img_input =  '/home/sarvagya/Desktop/RBC/Praful_code/rf_data/11.png'

img = cv2.imread(img_input)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

h_hls = img_hls[:,:,0]
l_hls = img_hls[:,:,1]
s_hls = img_hls[:,:,2]


h_hsv = img_hsv[:,:,0]
s_hsv = img_hsv[:,:,1]
v_hsv = img_hsv[:,:,2]

thresh_h = [120,180]
thresh_l = [150,250]
thresh_s = [0,20]

img_b = img[:,:,0]
img_g = img[:,:,1]
img_r = img[:,:,2]

binary_h_hls = np.zeros_like(h_hsv)
binary_l_hls = np.zeros_like(h_hsv)
binary_s_hls = np.zeros_like(h_hsv)


binary_l_hls[(l_hls > thresh_l[0]) & (l_hls < thresh_l[1])] = 255
binary_l_hls[:binary_l_hls.shape[0]//2,:]=0
binary_h_hls[(h_hls > thresh_h[0]) & (h_hls < thresh_h[1])] = 255
binary_h_hls[:binary_h_hls.shape[0]//2,:]=0
binary_s_hls[(s_hls > thresh_s[0]) & (s_hls < thresh_s[1])] = 255
binary_s_hls[:binary_s_hls.shape[0]//2,:]=0



result = np.bitwise_and(binary_l_hls,binary_h_hls)
result2 = np.add(result, binary_s_hls)

histogram = np.sum(result2[result2.shape[0]//2:,:], axis=0)

# plt.plot(histogram)
# plt.show()
# plt.imshow(result2)
# plt.show()

# f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(binary_h_hsl)
# ax1.set_title('h_hsl', fontsize=30)
# ax2.imshow(binary_l_hls)
# ax2.set_title('l_hls', fontsize=30)
# ax3.imshow(result)
# ax3.set_title('result', fontsize= 30)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

f, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(result2)
ax1.set_title('result', fontsize=30)
ax2.plot(histogram)
ax2.set_title('histogram', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
