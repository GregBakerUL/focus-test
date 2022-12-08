# import the necessary packages
from imutils import paths
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics
import sys
import csv

# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian
def variance_of_laplacian(image):

	return cv2.Laplacian(image, cv2.CV_64F).var()


# [1] Read image
# [2] Convert to grayscale
# [3] Replace embedded line row with previous row
# [4] Split image into left and right
# [5] Calculate laplacian for both images
# [6] Return results
def process_side_by_side_image(image_path):
    
    image = cv2.imread(image_path)                                      # [1]
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                # [2]
    image_gray[-1, :] = image_gray[-2, :]                               # [3]
    split_images = np.hsplit(image_gray, 2)                             # [4]
    
    left_laplacian_variance = variance_of_laplacian(split_images[0])    # [5]
    right_laplacian_variance = variance_of_laplacian(split_images[1])   # [5]
    
    return [left_laplacian_variance, right_laplacian_variance]          # [6] 


def display_image(image_path, focus_metric, label):
    image = cv2.imread(image_path) 
    cv2.putText(image, "LapVar: {:.2f}".format(focus_metric[0]), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 3)
    cv2.putText(image, "LapVar: {:.2f}".format(focus_metric[1]), (522, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 3)
    
    cv2.imshow(label, image)
    
    key = cv2.waitKey(0)
    

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
        help="path to input directory of images")
    ap.add_argument("-t", "--threshold", type=float, default=175.0,
        help="focus measures that fall below this value will be considered 'blurry'")
    args = vars(ap.parse_args())

    side_label = ["left", "right"]
    labels= []
    fm_av = [],[],[]
    fm_av_sensor_left = []
    fm_av_sensor_right = []
    
    display_images = False
    

    for image_path in paths.list_images(args["images"]):
        
        image_path_split = image_path.split('\\')
        if not (image_path_split[9] == 'Assembled' and image_path_split[10] == 'dome_focus'):
            continue
        
        label = image_path.split('\\')[8].split('FFFF0000')[1]
        if label not in labels:
            if len(labels) > 0:
                fm_av[0].append(labels[-1])
                fm_av[1].append(statistics.mean(fm_av_sensor_left))
                fm_av[2].append(statistics.mean(fm_av_sensor_right))
                
                
            print(label)
            labels.append(label)
            fm_av_sensor_left = []
            fm_av_sensor_right = []
        
        fm = process_side_by_side_image(image_path)
        if display_images and max(fm) < args["threshold"]:
            display_image(image_path, fm, label)
        fm_av_sensor_left.append(fm[0])
        fm_av_sensor_right.append(fm[1])      
    
    fm_av[0].append(labels[-1])
    fm_av[1].append(statistics.mean(fm_av_sensor_left))
    fm_av[2].append(statistics.mean(fm_av_sensor_right))
    

    # opening the csv file in 'w+' mode
    file = open('output.csv', 'w+', newline ='')
    
    with file:   
        write = csv.writer(file)
        write.writerows(fm_av)

    x_scatter = np.linspace(1, len(fm_av[0]), len(fm_av[0]))

    fig, ax = plt.subplots(1, 1, figsize=(10,9))
    plt.xlabel('Sensor')
    plt.ylabel('Focus [a.u.]')
    plt.title('Sensor focus level')
    plt.xticks(x_scatter, fm_av[0])
    plt.xticks(rotation=90)
    ax.scatter(x_scatter, fm_av[1], s=8, label= "Left")
    ax.scatter(x_scatter, fm_av[2], s=8, label="Right")
    ax.legend()
    ax.grid(True)
    fig.savefig("plots/scatter.png")
    
    
    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=fm_av[1], ax=ax1, label="Left")
    sns.kdeplot(data=fm_av[2], ax=ax1, label="Right")
    ax1.legend()
    fig1.savefig("plots/distribution.png")
    plt.show()
    
if __name__ == "__main__":
    main()