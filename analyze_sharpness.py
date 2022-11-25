# import the necessary packages
from imutils import paths
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np 

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

blur_array= []
labels= []

for imagePath in paths.list_images(args["images"]):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    print(imagePath)
    #words_tkn = imagePath.split('LE3100040000')
    words_tkn = imagePath.split('LE3100000000')
    words_tkn = words_tkn[1].split('_')
    lr_tkn= words_tkn[2].split('.png')
    #print(words_tkn)
    mylabel= words_tkn[0] + "_" + lr_tkn[0]
    
    labels.append(mylabel)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    #if fm < args["threshold"]:
    #	text = "Blurry"
    # show the image
    #cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
    #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #cv2.imshow("Image", image)
    #key = cv2.waitKey(0)
    blur_array.append(fm)
    print(fm)
    
x_scatter = np.linspace(1, len(blur_array), len(blur_array))
#labels = ["a", "b", "c", "d", "e", "6", "g", "h", "i", "l"]

fig, ax = plt.subplots(1, 1, figsize=(10,9))
plt.xlabel('Sensor')
plt.ylabel('Focus [a.u.]')
plt.title('Sensor focus level')
plt.xticks(x_scatter, labels)
plt.xticks(rotation=90)
#plt.tight_layout()
ax.scatter(x_scatter, blur_array, s=8, label= "T")
#ax.legend()
ax.grid(True)
plt.show()