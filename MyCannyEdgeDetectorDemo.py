import numpy as np
from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import feature #for final comparison
from math import log10,sqrt
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def conv_t(image):
  img_c=image.copy()
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      img_c[i][j]=image[image.shape[0]-1-i][image.shape[1]-j-1]
  return img_c

def conv(image,kernel):
  # kernel=conv_t(kernel)
  img_a=image.shape[0]
  img_b=image.shape[1]
  k_a=kernel.shape[0]
  k_b=kernel.shape[1]
  h=k_a//2
  w=k_b//2
  img_conv=np.zeros(image.shape)
  for i in range(h,img_a-h):
    for j in range(w,img_b-w):
      sum=0
      for m in range(k_a):
        for n in range(k_b):
          sum=sum+kernel[m][n]*image[i-h-k_a//2+m][j-w-k_b//2+n]
      img_conv[i][j]=sum
  return img_conv

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Ix = conv(img, Kx)
    Iy = conv(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 1
    return G
def sobel_filters1(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = conv(img, Kx)
    Iy = conv(img, Ky)
    theta = np.arctan2(Iy, Ix)
    
    return theta

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(0,M):
        for j in range(0,N):
            try:
                q = 255
                r = 255
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i][j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    return Z
def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(0, M):
        for j in range(0, N):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def myCanny(img, lowThresholdRatio, highThresholdRatio):
    img=non_max_suppression(sobel_filters(img),sobel_filters1(img)) 
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return hysteresis(res, weak, strong)

def myCannyEdgeDetector(img, lowThresholdRatio, highThresholdRatio):
    img= rgb2gray(img)
    k = gaussian_kernel(3,1)
    img = conv(img,k)
    img1= feature.canny(img)
    img= myCanny(img,0.07,0.11)
    print("Psnr value:",PSNR(np.array(img,dtype = bool),img1))
    score = ssim(np.array(img,dtype = bool), img1)
    print("ssim value:",score)
    fig, axes = plt.subplots(1, ncols=2, figsize=(16, 8))
    axes[0].imshow(img,cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(img1,cmap='gray')
    axes[1].axis('off')
    plt.show()


inputImg = data.astronaut()
myCannyEdgeDetector(inputImg,0.07,0.11)
# inputImg1 = rgb2gray(inputImg)
# k = gaussian_kernel(3,1)
# inputImg1 = conv(inputImg1,k)
# # inputImg2= sobel_filters(inputImg1)
# # img1=ndimage.convolve(inputImg,k)
# # print( ssim(inputImg1, img1))
# # inputImg3= sobel_filters1(inputImg1)
# # inputImg4=non_max_suppression(inputImg2,inputImg3)
# # print(np.median(inputImg4))
# # k=np.median(inputImg4)
# # lower = int(max(0, (1.0 - 0.33) * k))
# # upper = int(min(255, (1.0 + 0.33) * k))
# inputImg5= 


# print("Psnr value:",PSNR(np.array(inputImg5,dtype = bool),img))
# score = ssim(np.array(inputImg5,dtype = bool), img)
# print("ssim value:",score)
# fig, axes = plt.subplots(1, ncols=2, figsize=(16, 8))
# axes[0].imshow(inputImg5,cmap='gray')
# axes[0].axis('off')
# axes[1].imshow(img,cmap='gray')
# axes[1].axis('off')
# plt.show()
