"""
Importing all necessary packages
"""
import argparse
import numpy as np
import os
import cv2

def edge(img):
    """
    Returns the edges of the image (Canny detection)
    """
    image = cv2.Canny(img, 100, 200)

    return image

def hist_eq(img):
    """
    Returns the adaptive histogram equalization of an grayscale image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting BRG image to Grayscale image
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (5, 5))
    clahe_output = clahe.apply(img_gray)

    return clahe_output

def mask_image(img):
    """
    Returns the mask of the input image with black background
    """
    m = np.zeros(img.shape[:2], dtype = "uint8") # returns pure black image
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 70, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask = m) # masking the coin image with the black image

    return masked_img

def preprocess(img):
    """
    Return the circles detected in an image by Hough Circle Transform
    """
    fraction = 300/max(img.shape[0:2])
    img_resize = cv2.resize(img, None, fx = fraction, fy = fraction) # Resizing the image
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (5, 5)) # Performs blurring with a kernel size (5,5)

    coins = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1.2, 30, param2=35, minRadius=20, maxRadius=60) # Performs Hough circle transform
    coins = (np.round(coins[0, :]) / fraction).astype("int") # Rounding the decimal values to integer

    return coins

if __name__ == '__main__':
    """
    Using command line arguments for inputting folder name,
     coin denomination, format, szie and outout path of a file 
    """
    parser = argparse.ArgumentParser(description = "User input for image preprocessing")
    parser.add_argument("--folder", help = "Input image name")
    parser.add_argument("--coin_type", type = str, help = "Input image name")
    parser.add_argument("--format", default = "png", help = "output image format")
    parser.add_argument("--size", default = "150", type = int, help = "dimension of output images")
    parser.add_argument("--output_file", help="output Destination filename")
    args = parser.parse_args()

    file_name = os.listdir(args.folder) # Reading the folder name in a given file path

    for i,image in enumerate(file_name):
        img = cv2.imread(args.folder +"/"+ image) # Reads images one by one  in a folder

        coins = preprocess(img)
        radius = np.amax(coins, 0)[2] # Returns the maximum in an array along the rows

        for coin, (x, y, r) in enumerate(coins):
            each_coin = img[y - radius:y + radius, x - radius:x + radius] # Cropping out each coins detected in an image
            if each_coin.shape[0] ==0 or each_coin.shape[1] == 0: # Not taking the False positive coins detected in the image
                pass
            else:
                each_coin = cv2.resize(each_coin, (args.size, args.size))
                coin_name = "{}_{}_{}.{}".format(i, args.coin_type, coin, args.format) # Gives different name for each image
                output_path = os.path.join(args.output_file, coin_name)

                # Preprocessing of the image
                each_coin_mask = mask_image(each_coin) # Return mask of the input image with black background
                each_coin = hist_eq(each_coin_mask) # Adaptive Histogram equalization
                each_coin = edge(each_coin) # Finding Edges

                cv2.imwrite(output_path,each_coin) # Saves the image in a given folder path




