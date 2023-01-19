#%%
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from imutils import paths
import seaborn as sns
import argparse


class LaplacianVarianceImage():

    def __init__(self, image, name):
        self.image = image
        self.laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
        self.name = name
        self.laplacian_variance = self.laplacian_variance()
        
    def display_image(self):
        plt.imshow(self.image, cmap='gray')
        plt.title(self.name)
        plt.show()
    
    def laplacian_variance(self):
        return self.laplacian.var()
    
    def display_laplacian_image(self):
        plt.imshow(self.laplacian, cmap='gray')
        plt.title(self.name)
        plt.show()
        
    def display_laplacian_histogram(self):
        bins = np.linspace(-50, 50, 100)
        plt.hist(self.laplacian.flatten(), bins=bins)
        plt.xlim([-50, 50])
        plt.title(self.name)
        plt.show()
    
    
class LaplacianVarianceGroup():
    
    def __init__(self, image_dir, name="", stereo=False):
        self.image_dir = image_dir
        self.name = name
        self.image_paths = paths.list_images(image_dir)
        self.images = []
        if stereo:
            for path in self.image_paths:
                images = self.split_stereo_image(path)
                image_name = path.split(os.path.sep)[-1].split(".")[0]
                image_name_left = image_name + "_left"
                image_name_right = image_name + "_right"
                self.images.append(LaplacianVarianceImage(images[0], image_name_left))
                self.images.append(LaplacianVarianceImage(images[1], image_name_right))
        else:
            for path in self.image_paths:
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                image_name = path.split(os.path.sep)[-1].split(".")[0]
                self.images.append(LaplacianVarianceImage(image, image_name))
        self.images_sorted = self.sort_by_focus()
    
    def split_stereo_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images = np.hsplit(image, 2)
        return images

    def sort_by_focus(self, reverse=False):
        return sorted(self.images, key=lambda image: image.laplacian_variance, reverse=reverse)
    
    def plot_focus_scatter(self, ax=None, plot=True):
        if ax is None:
            fig, ax = plt.subplots()
            plot = True
        x = [image.name for image in self.images_sorted]
        y = [image.laplacian_variance for image in self.images_sorted]
        ax.scatter(x, y)
        ax.set_xticklabels(x, rotation=90)
        if plot:
            plt.show()
        return ax
    
    def plot_focus_kde(self, ax=None, plot=True):
        if ax is None:
            fig, ax = plt.subplots()
        y = [image.laplacian_variance for image in self.images_sorted]
        sns.kdeplot(y, ax=ax)
        if plot:
            plt.show()
        return ax
        
class LaplacianVarianceGroupComparison():
    
    def __init__(self):
        self.image_groups = []
        
    def add_image_group(self, image_group):
        self.image_groups.append(image_group)
        
    def add_image_dir(self, image_dir, name="", stereo=False):
        image_group = LaplacianVarianceGroup(image_dir, name=name, stereo=stereo)
        self.image_groups.append(image_group)
    
    def plot_focus_scatter(self, ax=None, plot=True):
        if ax is None:
            fig, ax = plt.subplots()
        for image_group in self.image_groups:
            image_group.plot_focus_scatter(ax=ax, plot=False)
        if plot:
            ax.legend([image_group.name for image_group in self.image_groups])
            plt.show()
        return ax
    
    def plot_focus_kde(self, ax=None, plot=True):
        if ax is None:
            fig, ax = plt.subplots()
        for image_group in self.image_groups:
            image_group.plot_focus_kde(ax=ax, plot=False)
        if plot:
            ax.legend([image_group.name for image_group in self.image_groups])
            plt.show()
        return ax
    
    def print_focus_stats(self):
        for image_group in self.image_groups:
            print(image_group.name)
            for image in image_group.images_sorted:
                print(image.name, image.laplacian_variance)
            print("")
    
        
    
    

#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-path", help="path to an image file to be processed")
    parser.add_argument("-d", "--image-directory", help="path to a directory of images to be processed")
    args = parser.parse_args()
    
    if args.image_path is None and args.image_directory is None:
        raise ValueError("Please specify a path to an image directory or image file.")
    
    if args.image_path is not None:
        image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        image_name = args.image_path.split(os.path.sep)[-1].split(".")[0]
        laplacian_variance_image = LaplacianVarianceImage(image, image_name)
        laplacian_variance_image.display_image()
        laplacian_variance_image.display_laplacian_histogram()
    if args.image_directory is not None:
        image_group = LaplacianVarianceGroup(args.image_directory)
        image_group.plot_focus_scatter()
        image_group.plot_focus_kde()

    