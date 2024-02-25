"""
This file executes the non-textual data extraction system implemented for Formula 1 cars images. 
The program asks the user for an input image and then performs a comparison with the rest of the images in the dataset. 
The comparison is based on the similarity of the histograms of the images and the text found in the images. The program returns the 5 most similar images to the input image and plots them. 
It also plots the histograms of the 5 most similar images and prints the words found in the images.

Usage:  python non_textual_data_extraction.py

Input:  The user is asked to introduce a number between 0 and the number of images in the dataset. 
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import subprocess
subprocess.call('info_retrieval', shell=True)

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from functions import *

def load_data():
    cars_df = pd.DataFrame(columns=['car_id', 'path'])
    cars_df = get_images_paths("data\Formula_one_cars\Alphatauri", cars_df)
    cars_df = get_images_paths("data\Formula_one_cars\Ferrari", cars_df)
    cars_df = get_images_paths("data\Formula_one_cars\Mclaren", cars_df)
    cars_df = get_images_paths("data\Formula_one_cars\Mercedes", cars_df)
    cars_df = get_images_paths("data\Formula_one_cars\Racingpoint", cars_df)
    cars_df = get_images_paths("data\Formula_one_cars\Redbull", cars_df)
    cars_df = get_images_paths("data\Formula_one_cars\Renault", cars_df)
    cars_df = get_images_paths("data\Formula_one_cars\Williams", cars_df)
    return cars_df

def get_query_image(cars_df):
    # Ask user for an input image  - ask for a number between 0 and len(cars_df)
    print("Please, introduce a number between 0 and ", len(cars_df), "(e.g. 20)")
    number = int(input())
    plt.figure(figsize=(10,10))
    query_img_path = cars_df['path'][number]
    query_img = Image.open(query_img_path)
    plt.imshow(query_img)
    plt.title('Query Image. Number introduced: ' + str(number))
    plt.show()
    return query_img_path

def get_query_image_descriptors(query_img_path, th = 200):
    # Get descriptors
    query_text = ocr(query_img_path)
    print("Text found in query image: ", query_text)
    hist = histogram(query_img_path, th, plot=True)
    return query_text, hist

def plot_results(values_df_sorted, n=5):
    # Plot images
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(Image.open(values_df_sorted['car_id'].iloc[i]))
        plt.title('Similarity ' + str(round(values_df_sorted['similarity_value'].iloc[i], 3)))
    plt.show()
    
    # Plot histogram
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        hist = histogram(values_df_sorted['car_id'].iloc[i], th=200, plot=False)
        plt.plot(hist[0], color='r')
        plt.plot(hist[1], color='g')
        plt.plot(hist[2], color='b')
        plt.title('Image ' + str(i+1))
    plt.show()
    
    # Print words
    for i in range(5):
        text = ocr(values_df_sorted['car_id'].iloc[i])
        print('words in image ', i+1, ': ', text)
        print('-----------------------------------')

    

def main():
    # Load data
    cars_df = load_data()
    
    # Query Image
    query_img_path = get_query_image(cars_df)
    
    # Get descriptors
    query_text, hist = get_query_image_descriptors(query_img_path, th = 200)
    
    # Perform comparation with the rest of the images
    values_df, results_df = perform_comparation(query_img_path, cars_df, n1=0.5, n2 = 0.5)
    values_df = values_df.sort_values(by='similarity_value', ascending=False)
    
    plot_results(values_df)
    

if __name__ == "__main__":
    main()
