import cv2 
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow


def partone():
    im=cv2.imread('10.png',cv2.IMREAD_GRAYSCALE)

    f=np.fft.fft2(im)
    fshift=np.fft.fftshift(f)
    magnitude= 20*np.log(np.abs(fshift))

    magnitude1=np.asarray(magnitude,dtype=np.uint8)

    cv2_imshow(im)
    cv2_imshow(magnitude1)
    print("\n")
    cv2_imshow(magnitude)


def filerimage():
  im=cv2.imread('15.jpg',0)
  dft=cv2.dft(np.float32(im),flags=cv2.DFT_COMPLEX_OUTPUT)
  dft_shi=np.fft.fftshift(dft)
  magmit=20*np.log(cv2.magnitude(dft_shi[:,:,0],dft_shi[:,:,1]))
  #mask
  row,cols=im.shape
  crow,ccol=int(row/2),int(cols/2)
  mask=np.ones((row,cols,2),np.uint8)
  r=60
  cnter=[crow,ccol]
  x,y=np.ogrid[:row,:cols]
  maskarea=(x-cnter[0])**2+(y-cnter[1])**2 <= r*r
  mask[maskarea]=0
  fshift=dft_shi*mask
  fshift_mask_mag=2000*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
  f_isift=np.fft.ifftshift(fshift)
  img_black=cv2.idft(f_isift)
  img_black=cv2.magnitude(img_black[:,:,0],img_black[:,:,1])
  plt.imshow(img_black,cmap='gray')
  plt.show()
  fig, axs = plt.subplots(2, 2)
  axs[0, 0].imshow(im,cmap='gray')
  
  axs[0, 1].imshow(magmit,cmap='gray')
  
  axs[1, 0].imshow(fshift_mask_mag,cmap='gray')
  
  axs[1, 1].imshow(img_black,cmap='gray')
  
  fig.show()
  

#partone() 
filerimage()