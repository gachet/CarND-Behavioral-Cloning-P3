import numpy as np
import cv2


def speckle_noise(image, *args):
    gaussian = np.random.randn(*image.shape)
    #noisy = image + image * gaussian
    noisy = cv2.addWeighted(image.astype(np.float64), 0.9, image * gaussian, 0.05, 0.0)
    noisy = noisy.astype(np.uint8)
    return noisy


def salt_and_pepper(image, *args):
    salt_th = 253
    pepper_th = 3
    
    noise = np.random.randint(0, 256, image.shape[:2])
    salt_noise = noise > salt_th
    pepper_noise = noise < pepper_th
    
    result = image.copy()
    result[salt_noise] = 255
    result[pepper_noise] = 0
    
    return result


def bilateral_filter(image, *args):
    return cv2.bilateralFilter(image,9,75,75)


def horizontal_flipping(image):
    return np.fliplr(image)


def shift(image, *args):
    # We are going to allow a maximum shift of +-5
    if len(args):
        x_shift = args[0]
        y_shift = args[1]
        x_shift = np.clip(x_shift, -5, 5)
        y_shift = np.clip(y_shift, -5, 5)
    else:
        x_shift = np.random.uniform(-5, 5+1)
        y_shift = np.random.uniform(-5, 5+1)
    
    image_center = tuple(np.array(image.shape)/2)
    shift_mat = np.array([[1, 0, x_shift], [0, 1, y_shift]], dtype=np.float64)
    result = cv2.warpAffine(image, shift_mat, image.shape[:2], borderMode=cv2.BORDER_REFLECT)
    return result


def rotateImage(image, *args):
    # We are going to allow a maximum angle of +-15
    if len(args):
        angle = args[0]
        angle = np.clip(angle, -15, 15)
    else:
        angle = np.random.uniform(-15, 15+1)
    
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center[:2],angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], borderMode=cv2.BORDER_REFLECT_101)
    return result


def random_light(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.25+np.random.uniform()
    random_bright = np.clip(random_bright, 0.25, 1.0) # Values greater than 1.0 produces unwanted noise
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
          
    
def random_shadow(image):
    top_y = image.shape[1]*np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1]*np.random.uniform()
    image_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    shadow_mask = 0*image_hsv[:,:,2]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1

    random_bright = .4+np.random.uniform()
    random_bright = np.clip(random_bright, 0.4, 0.9) # Between 0.4 and 0.9 is the ideal value to generate shadows
    
    cond1 = shadow_mask==1
    cond0 = shadow_mask==0

    if np.random.randint(2)==1:
        image_hsv[:,:,2][cond1] = image_hsv[:,:,2][cond1]*random_bright
    else:
        image_hsv[:,:,2][cond0] = image_hsv[:,:,2][cond0]*random_bright  
            
    image = cv2.cvtColor(image_hsv,cv2.COLOR_HSV2RGB)

    return image


def histogram_equalization(image):
    """
    http://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
    the first step is to convert the color space of the image from RGB into one of the color space which separates intensity values from color components.
    Perform HE of the intensity plane.
    """
    image1 = cv2.cvtColor(image,cv2.CV_RBG2YCrCb)
    image1[:, :, 0] = cv2.equalizeHist(image1[:, :, 0])
    return cv2.cvtColor(image,cv2.CV_YCrCb2RBG)


def identity(image):
    return image
    
    
transformations = [rotateImage, shift, bilateral_filter, salt_and_pepper, speckle_noise, horizontal_flipping, \
                   random_light, random_shadow, histogram_equalization, identity]