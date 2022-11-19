from skimage import data
from skimage.color import rgb2gray
from imageio import imread
import numpy as np
import sys

def conv_t(image):
  img_c=image.copy()
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      img_c[i][j]=image[image.shape[0]-1-i][image.shape[1]-j-1]
  return img_c
def conv(image,kernel):
  kernel=conv_t(kernel)
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
          sum=sum+kernel[m][n]*image[i-h-m][j-w-n]
      img_conv[i][j]=sum
  return img_conv

  
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def laplacian(img):
    Kx = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    Ix = conv(img, Kx)
    return Ix.var()
    
def prob(k_m,k):
    return (k/k_m)

def BlurOrNot(img): 
  img=rgb2gray(img)
  # img=conv(img,gaussian_kernel(3,1))
  k=laplacian(img)
  print(k)
  k_m=	0.11038868993908181
  k_me= 0.0085654738086132
  k_m=max(k,k_m)
  p=prob(k_m,k)
  p1=prob(k_me,k)
  if(abs(k-k_m)<abs(k-k_me)):
    p2=1-p
  else:
    p2=p1-0.5
  print("probability",p2)
  if(k>	k_me):
    print("unblur")
  else:
    print("blur")

img=data.astronaut()
BlurOrNot(img)


# def vari(img):
#   l=img.shape[0]
#   b=img.shape[1]
#   psum=0;
#   for i in range(1,l-1):
#     for j in range(1,b-1):
#     #  try:
#       sum=0;
#       sum+=pow((img(i+1,j+1)-img(i,j))/255,2)
#       sum+=pow((img(i,j+1)-img(i,j))/255,2)
#       sum+=pow((img(i+1,j)-img(i,j))/255,2)
#       sum+=pow((img(i-1,j-1)-img(i,j))/255,2)
#       sum+=pow((img(i,j-1)-img(i,j))/255,2)
#       sum+=pow((img(i-1,j+1)-img(i,j))/255,2)
#       sum+=pow((img(i+1,j-1)-img(i,j))/255,2)
#       sum+=pow((img(i-1,j)-img(i,j))/255,2)
#       sum/=8
#       psum+=sum
#     #  except:
#     #   next
#   psum/=(l-1)*(b-1)
#   return psum