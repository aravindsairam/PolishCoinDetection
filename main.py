"""
Importing all necessary packages
"""
import os
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_processing import preprocess, mask_image, hist_eq, edge


def load_data(img, size):
    """
    Return the image in grayscale,
    with an array shape of (size*size,)
    """
    img =  cv2.imread(img, 0)
    img = cv2.resize(img, (size, size))
    img = img.flatten()

    return img

def normalize_array(img):
    """
    Returns an Normalized image array
    by dividing the array by 255
    """
    img =img/255

    return img


def predict_result(predict):
    """
    Returns the predicted coin denomination names
    corresponding to the neural network output
    """
    result = {0:'one_grosz', 1:'two_grosz', 2:'five_grosz', 3:'ten_grosz',
              4:'twenty_grosz', 5:'fifty_grosz', 6:'one_zloty', 7:'two_zloty', 8:'five_zloty'}

    return result[predict]

def addition(result):
    """
    Returns the value of the given coin denominations
    """
    values = {'one_grosz':0.01 , 'two_grosz':0.02, 'five_grosz':0.05, 'ten_grosz':0.10,
              'twenty_grosz':0.20, 'fifty_grosz':0.50, 'one_zloty':1.00, 'two_zloty':2.00, 'five_zloty':5.00}

    return values[result]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "User input for image preprocessing")
    parser.add_argument("--file_path", help = "Training dataset")
    parser.add_argument("--size", default = "64", type=int, help="dimension of output images")
    parser.add_argument("--input_image", help = " Image for prediction")
    args = parser.parse_args()

    h, w = args.size, args.size # All the images are interpreted in array of shape (64*64)

    X = []
    y = []

    files_name = os.listdir(args.file_path) # Getting all the folder names in a path

    temp_name = ['1_grz', '2_grz', '5_grz', '10_grz', '20_grz', '50_grz', '1_zl', '2_zl', '5_zl']

    for n, _ in enumerate(files_name):
        temp = glob.glob(args.file_path +"/"+ temp_name[n]+"/*")

        for i in temp:
            X.append(load_data(i, args.size)) # Loading all the images with a shape of (64*64)
            y.append(int(n)) # Labels for the input images

    # Converting list to arrray
    X = np.asarray(X, dtype = np.float32)
    y = np.asarray(y, dtype = np.float32)

    X = normalize_array(X) # Normalizing the image data

    # Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True,
                                                        random_state = 42, test_size = 0.1)

    # Initializing the Neural Network model
    """
    Created a 3 layer Neural network with 2 hidden layers.
    Size of hidden layer 1 = 512
    Size of hidden layer 1 = 256
    Size of output layer = 9 ( nine different coin denominations)
    Dropout Regularization with keep probability 0.5 in hidden layer 1 and 0.7 in hidden layer 2
    hidden Layer 1 and 2 = ReLu (Retified Linear Unit) activation function used
    Output Layer = Softmax activation function
    """
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, kernel_initializer = tf.keras.initializers.he_normal(seed = None),
                            activation = tf.nn.relu),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(256, kernel_initializer = tf.keras.initializers.he_normal(seed = None),
                            activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.he_normal(seed=None),
        activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.7),
      tf.keras.layers.Dense(9, activation = tf.nn.softmax)
    ])

    """
    Optimized : Adam optimized
    Learning rate : 0.001
    metrices : accuracy 
    """
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trained with 5 epochs with the batch size of 32
    model.fit(X_train, y_train, epochs= 5 )

    # Evaluate the Testing set
    print(model.evaluate(X_test,y_test))


    # Prediction
    output = cv2.imread(args.input_image) # Loading the necessay image for prediction

    # Preproceesing of the image
    """
    Getting circles with Hough circle transform
    Histogram equalization and finding the egdes of the images
    Reshape and flatten the image  
    """
    inp_img = preprocess(output)
    radius = np.amax(inp_img, 0)[2] # Returns the maximum in an array along the rows

    total_sum = 0

    for coin, (x, y, r) in enumerate(inp_img):  # Looping over all the circles one by one
        each_coin = output[y - radius:y + radius, x - radius:x + radius]  # Cropping out each coins detected in an image

        if each_coin.shape[0] == 0 or each_coin.shape[1] == 0: # Not taking the False positive coins detected in the image
            pass
        else:
            each_coin = cv2.resize(each_coin, (args.size, args.size))
            each_coin = mask_image(each_coin)
            each_coin = hist_eq(each_coin)
            each_coin = edge(each_coin)

            temp_img = each_coin.flatten()
            temp_img = normalize_array(temp_img)
            temp_img = cv2.resize(temp_img, (1, h*w)).T

            predict_coin = model.predict([temp_img])  # Predicting the image with the trained neural network model

            result = predict_result(np.argmax(predict_coin[0]))



        total_sum += addition(result) # Getting the total sum of all the predicted coins
        cv2.circle(output, (x, y), r, (0, 255, 0), 2) # Drawing the circles with radius r in the image used for prediction
        cv2.putText(output, result,(x - 40, y), cv2.FONT_HERSHEY_PLAIN,5, (0, 255, 0),
                    thickness=2, lineType=cv2.LINE_AA) # Writing the predicted coin denominations on each coin of the image


    cv2.putText(output, "There is {} Zloty worth of coins".format(total_sum), (5, output.shape[0] - 8),
                cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA) # Writing the total sum in the image

    output = cv2.resize(output, None, fx = 1/2, fy= 1/2) # The loaded predicted image is very big so resizing by half the original resolution

    cv2.imshow('output' ,output) # Displays the predicted image
    cv2.waitKey(0)
    cv2.destroyAllWindows()





