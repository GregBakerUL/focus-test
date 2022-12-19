#%%
# import the necessary packages
from imutils import paths
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import seaborn as sns
import statistics
import pandas
import os


# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian
def variance_of_laplacian(image):

	return cv2.Laplacian(image, cv2.CV_64F).var()


def find_files_in_dir( path_to_dir, suffix=".png"):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


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
    cv2.putText(image, "LapVar: {:.2f}".format(focus_metric), (10, 30),
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

    side_label = ["L", "R"]
    labels= []
    fm_av = []
    fm_image = []
    counter = 0
    id = 1
    counters = []
    
    display_images = False

    for image_path in paths.list_images(args["images"]):
        image_name = image_path.split('\\')[-1]
        serial = image_name.split('.')[0].split('_')[2] + '_' + image_name.split('.')[0].split('_')[3]
        if serial not in labels:
            if len(labels) > 0:
                fm_av.append(statistics.fmean(fm_image))
                
            print(serial)
            labels.append(serial)

            # check if counter is odd or even
            side = side_label[counter % 2]
            if counter % 2 == 1:
                id += 1
            counters.append(str(id) + '_' + side)
            counter += 1
            fm_image = []
        
        image = cv2.imread(image_path)
        fm = variance_of_laplacian(image)
        
        if display_images and fm < args["threshold"]:
            display_image(image_path, fm, serial)
        fm_image.append(fm)

    fm_av.append(statistics.fmean(fm_image))
    
    fm_av_df = pandas.DataFrame(fm_av)
    # fm_av_df = fm_av_df.transpose()
    fm_av_df.index = counters
    fm_av_df.sort_values(by=fm_av_df.columns[0], inplace=True)
    print(fm_av_df.head())
    

    x_scatter = np.linspace(1, len(fm_av_df), len(fm_av_df))

    fig, ax = plt.subplots(1, 1, figsize=(15,9))
    plt.xlabel('Sensor')
    plt.ylabel('Focus [a.u.]')
    plt.title('Sensor focus level')
    plt.xticks(x_scatter, fm_av_df.index, rotation=90)
    ax.scatter(x_scatter, fm_av_df, s=8)
    ax.legend()
    ax.grid(True)
    fig.savefig("plots/scatter.png")
    
    
    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=fm_av, ax=ax1, label="Focus")
    ax1.legend()
    fig1.savefig("plots/distribution.png")
    plt.show()
    
if __name__ == "__main__":
    main()
# %%
