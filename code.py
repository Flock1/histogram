import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas

#img = cv2.imread("for_assignment/DSC01798.jpg")
img = cv2.imread('/home/sarvagya/Desktop/RBC/Praful_code/rf_data/0.png')

b_colour = img[:,:,0]  #blue colour gradient
g_colour = img[:,:,1]  #green colour gradient

#Use the blue and h and l to detect the highest and common values and find the common

#Search for something that helps with pixel intensity of density or something

def process_img(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hsv = img_hsv[:,:,0]
    s_hsv = img_hsv[:,:,1]
    v_hsv = img_hsv[:,:,2]

    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_hls = img_hls[:,:,0]
    l_hls = img_hls[:,:,1]
    s_hls = img_hls[:,:,2]

    # with open('for_assignment/gcp_positions.csv', 'rb') as gcp_code:
    #     reader = csv.reader(gcp_code, delimiter=' ', quotechar='|')
    #
    # print(np.shape(img))
    # for row in reader:
    #     print(", ".join(reader))

    # #################################### TO READ CSV FILE ###########################################
    # df = pandas.read_csv("for_assignment/gcp_positions.csv", sep=",")
    #
    # rows, columns = df.shape[0], df.shape[1]
    #
    #
    #
    # print(df['GCPLocation'][0])
    #
    # image_1_x = int(float(df['GCPLocation'][0].split()[0][2:-1]))
    # image_1_y = int(float(df['GCPLocation'][0].split()[1][:-2]))
    #
    # print(np.float64(df['GCPLocation'][0].split()[0][2:-1]))  #Gets the first coordinate from the file
    # print(np.float64(df['GCPLocation'][0].split()[1][:-2]))   #Gets the second coordimate from the file
    #
    # img_laplace = cv2.Sobel(img, cv2.CV_32F, 0,1,ksize=15)
    #################################### TO READ CSV FILE ###########################################

    thresh_1 = [150,250]
    thresh_2 = [150,250]
    thresh_h = [130,180]


    # h_hsv[(h_hsv < thresh[0]) & (h_hsv > thresh[1])] = 0
    binary_l_hsl = np.zeros_like(h_hsv)
    binary_h_hsl = np.zeros_like(h_hsv)
    binary_s_hsl = np.zeros_like(h_hsv)
    binary_b_colour = np.zeros_like(b_colour)
    binary_g_colour = np.zeros_like(g_colour)
    half_point = binary_h_hsl//2

    binary_l_hsl[(l_hls > thresh_1[0]) & (l_hls < thresh_1[1])] = 255
    binary_l_hsl[:binary_l_hsl.shape[0]//2-50,:]=0
    binary_h_hsl[(h_hls > thresh_h[0]) & (h_hls < thresh_h[1])] = 255
    binary_h_hsl[:binary_h_hsl.shape[0]//2-50,:]=0
    # print(l_hls.max())
    # print(l_hls.min())
    # binary_s_hsl[(s_hls > thresh_1[0]) & (s_hls < thresh_1[1])] = 1
    # binary_b_colour[(b_colour > thresh_2[0]) & (b_colour < thresh_2[1])] = 1
    # binary_g_colour[(g_colour > thresh_2[0]) & (g_colour < thresh_2[1])] = 1

    result = np.bitwise_and(binary_l_hsl,binary_h_hsl)

    print(h_hls.max(), h_hls.min())
    return binary_l_hsl
# cv2.circle(img_hls,(image_1_x,image_1_y), 25, (255,0,0), 1)
# plt.imshow(v_hsv)
# plt.show()
# cv2.waitKey(0)
#hist = cv2.calcHist([img])

# f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(result)
# ax1.set_title('result', fontsize=30)
# ax2.imshow(l_hls)
# ax2.set_title('l_hls', fontsize=30)
# ax3.imshow(h_hls)
# ax3.set_title('h_hls', fontsize= 30)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

# cv2.destroyAllWindows()

#y length needs to 30 & x should be 20
