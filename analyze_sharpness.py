# import the necessary packages
from imutils import paths
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
        help="path to input directory of images")
    ap.add_argument("-t", "--threshold", type=float, default=100.0,
        help="focus measures that fall below this value will be considered 'blurry'")
    args = vars(ap.parse_args())

    blur_array= []
    blur_array_right= []
    labels= []
    show_images = True

    for image_path in paths.list_images(args["images"]):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        # print(image_path)
        image_path_split = image_path.split('\\')
        if not (image_path_split[9] == 'Assembled' and image_path_split[10] == 'dome_focus' and '000.png' in image_path_split[11]):
            continue
        
        label = image_path.split('\\')[8].split('FFFF0000')[1]
        print(label)
        
        labels.append(label)
        image = cv2.imread(image_path)
        
        # Replace embedded line (final row) with penultimate row as it 
        # has very strong gradients that throw the laplacian
        image[-1, :] = image[-2, :] 
        radius2 = 150
        height = image.shape[0]
        width = image.shape[1] // 2
        xc = height // 2
        yc = width // 2
        
        
        left = image[:, :width]
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        # Idea to circular crop image for unglued images to remove
        # LED artifact at the bottom
        # Currently not working well as it introduces a sharp gradient
        mask = np.zeros(left.shape[:2], dtype="uint8")
        cv2.circle(mask, (xc,yc), radius2, 255, -1)
        left_gray_crop = cv2.bitwise_and(left_gray, left_gray, mask=mask)
        left_fm = variance_of_laplacian(left_gray)
        
        right = image[:, width:]
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        right_gray_crop = cv2.bitwise_and(right_gray, right_gray, mask=mask)
        right_fm = variance_of_laplacian(right_gray)
        
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        #if fm < args["threshold"]:
        #	text = "Blurry"
        # show the image
        if show_images:
            cv2.putText(left_gray, "{:.2f}".format(left_fm), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 3)
            cv2.imshow("left", left_gray)
            
            cv2.putText(right_gray, "{:.2f}".format(right_fm), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 3)
            cv2.imshow("right", right_gray)
            
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                show_images = False
        
        blur_array.append(left_fm)
        blur_array_right.append(right_fm)
        print(left_fm)
        print(right_fm)
        
    x_scatter = np.linspace(1, len(blur_array), len(blur_array))

    fig, ax = plt.subplots(1, 1, figsize=(10,9))
    plt.xlabel('Sensor')
    plt.ylabel('Focus [a.u.]')
    plt.title('Sensor focus level')
    plt.xticks(x_scatter, labels)
    plt.xticks(rotation=90)
    ax.scatter(x_scatter, blur_array, s=8, label= "Left")
    ax.scatter(x_scatter, blur_array_right, s=8, label="Right")
    ax.legend()
    ax.grid(True)
    fig.savefig("plots/scatter.png")
    
    
    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=blur_array, ax=ax1, label="Left")
    sns.kdeplot(data=blur_array_right, ax=ax1, label="Right")
    ax1.legend()
    fig1.savefig("plots/distribution.png")
    plt.show()
    
if __name__ == "__main__":
    main()