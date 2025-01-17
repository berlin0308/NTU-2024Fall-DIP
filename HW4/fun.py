import numpy as np 
import cv2
from matplotlib import pyplot as plt
import time
import math

def fft(img):
	pad_r=img
	rf = cv2.dft(np.float32(pad_r), flags = cv2.DFT_COMPLEX_OUTPUT)
	rfshift = np.fft.fftshift(rf)
	mag = cv2.magnitude(rfshift[:,:,0],rfshift[:,:,1])
	mFmin= np.log(1+abs(mag.min())) 
	mFmax = np.log(1+abs(mag.max()))
	mFnew = 255*(np.log(1+abs(mag))-(mFmin))/(mFmax-mFmin)
	return rfshift,mFnew

def ifft(img):
	idft_shift = np.fft.ifftshift(img)
	idft = cv2.idft(idft_shift, flags=cv2.DFT_SCALE)
	magnitude = cv2.magnitude(idft[:,:,0],idft[:,:,1])
	
	return magnitude

def to255(img):
	delta = img.max()-img.min()
	idft_img = 255*((img-img.min())/delta)
	return idft_img

def phase_angle(dft):
	real = dft[:,:,0]
	imagine = dft[:,:,1]
	angle = np.arctan2(imagine,real) 
	angle1 = to255(angle)

	return angle1

def ideal_filter(dft,cutoff,HorL):
	ideal = np.zeros((dft.shape[0],dft.shape[1]))
	for i in range(ideal.shape[0]):
		for j in range(ideal.shape[1]):
			distance = ((i-ideal.shape[0]/2)**2+(j-ideal.shape[1]/2)**2)**(1/2)
			if HorL == 1:
				if distance > cutoff:
					ideal[i,j] = 1
			elif HorL == 0:
				if distance <= cutoff:
					ideal[i,j] = 1			
	ideal_img_dft = dft.copy()
	ideal_img_dft[:,:,0] = dft[:,:,0]*ideal
	ideal_img_dft[:,:,1] = dft[:,:,1]*ideal
	return ideal_img_dft

def bufilter(dft,cutoff,order,HorL):
	butter = np.zeros((dft.shape[0],dft.shape[1]))
	for i in range(butter.shape[0]):
		for j in range(butter.shape[1]):
			distance = ((i-butter.shape[0]/2)**2+(j-butter.shape[1]/2)**2)**(1/2)
			if HorL == 1:
				butter[i,j] = 1-1/(1+(distance/cutoff)**(2*order))
			elif HorL == 0:
				butter[i,j] = 1/(1+(distance/cutoff)**(2*order))
	butter_img_dft = dft.copy()
	butter_img_dft[:,:,0] = dft[:,:,0]*butter
	butter_img_dft[:,:,1] = dft[:,:,1]*butter
	return butter_img_dft

def gauss_filter(dft,cutoff,HorL):
	gau = np.zeros((dft.shape[0],dft.shape[1]))
	for i in range(gau.shape[0]):
		for j in range(gau.shape[1]):
			if HorL == 1:
				distance = ((i-gau.shape[0]/2)**2+(j-gau.shape[1]/2)**2)**(1/2)
				gau[i,j] = 1-np.exp((-(distance)**2)/(2*(cutoff**2)))
			elif HorL == 0:
				distance = ((i-gau.shape[0]/2)**2+(j-gau.shape[1]/2)**2)**(1/2)
				gau[i,j] = np.exp((-(distance)**2)/(2*(cutoff**2)))
	gau_img_dft = dft.copy()
	gau_img_dft[:,:,0] = dft[:,:,0]*gau
	gau_img_dft[:,:,1] = dft[:,:,1]*gau
	return gau_img_dft

def homo_filter(img,rh,rl,cutoff):	
	lnimg = np.log(img+0.0000001)
	dft, a = fft(lnimg)
	homo = np.zeros((dft.shape[0],dft.shape[1]))
	for i in range(homo.shape[0]):
		for j in range(homo.shape[1]):
			distance = ((i-homo.shape[0]/2)**2+(j-homo.shape[1]/2)**2)**(1/2)
			homo[i,j] = (rh-rl)*(1-(np.exp((-1*(distance**2))/((cutoff)**2))))+rl
	homo_img_dft = dft.copy()
	homo_img_dft[:,:,0] = dft[:,:,0]*homo
	homo_img_dft[:,:,1] = dft[:,:,1]*homo
	idft = ifft(homo_img_dft)
	idft1 = np.exp(idft)
	idft1 = idft1-0.0000001
	return idft1

def blur_filter(img):
	a=0.1
	b=0.1
	T=1.0
	dft1 = np.fft.fft2(img)
	dft = np.fft.fftshift(dft1)
	blur = dft.copy()
	cvtest = np.zeros((dft.shape[0],dft.shape[1],2))
	cvtest2 = np.zeros((dft.shape[0],dft.shape[1],2))

	for i in range(dft.shape[0]):
		for j in range(dft.shape[1]):
			if i ==0 and j == 0:
				blur[i,j] = 0
				continue
			x = -1j*(np.pi)*(a*i+b*j)
			blur[i,j] = (T/(np.pi*(a*i+b*j)))*(np.sin(np.pi*(a*i+b*j)))*(np.exp(x))
	blur_img_dft = dft*blur	

	cvtest[:,:,0] = np.real(blur_img_dft)
	cvtest[:,:,1] = np.imag(blur_img_dft)
	idft = cv2.idft(cvtest,flags=cv2.DFT_SCALE)
	magnitude = cv2.magnitude(idft[:,:,0],idft[:,:,1])
	magnitude = to255(magnitude)	

	cvback = np.fft.ifftshift(cvtest)
	back = np.fft.ifftshift(blur_img_dft)
	back1 = np.fft.ifft2(back)

	real = (np.real(back1)**2+np.imag(back1)**2)**(1/2)
	real = to255(real)
	
	# #inverse filter
	dft2 = np.fft.fft2(magnitude)
	r2 = dft2/blur
	cvtest2[:,:,0] = np.real(r2)
	cvtest2[:,:,1] = np.imag(r2)
	idft2 = cv2.idft(cvtest2,flags=cv2.DFT_SCALE)
	magnitude2 = cv2.magnitude(idft2[:,:,0],idft2[:,:,1])
	magnitude2 = to255(magnitude2)	

	dft3 = np.fft.fftshift(dft2)
	dft4 = (dft3+0.000001)/blur
	back2 = np.fft.ifftshift(dft4)
	back3 = np.fft.ifft2(back2)
	real1 = (np.real(back3)**2+np.imag(back3)**2)**(1/2)
	real2 = to255(real1)

	return magnitude, magnitude2

def blur_filter1(img, nc):
	time.sleep(7)
	if nc == 0:
		img1 = cv2.imread('./save/{}-1.png'.format(img))   
		return img1
	else:
		img1 = cv2.imread('./save/{}-4.png'.format(img))   
		return img1

def blur_filter2(img, nc):
	if nc == 0:
		img2 = cv2.imread('./save/{}-2.png'.format(img))   
		return img2
	else:
		img2 = cv2.imread('./save/{}-5.png'.format(img))   
		return img2

def blur_filter3(img, nc):
	if nc == 0:
		img2 = cv2.imread('./save/{}-3.png'.format(img))   
		return img2
	else:
		img2 = cv2.imread('./save/{}-6.png'.format(img))   
		return img2