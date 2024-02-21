"""
This file contains the main functions used in the Second Assignment.

The functions are:
    - get_images_paths: function to get the paths of the images in a folder and store them in a DataFrame
    - ocr: function to perform OCR on an image. It removes non-alphanumeric characters and returns a list of words in the image.
"""

import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

import re
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def get_images_paths(folder_path, df, n=200):
    """
    Function to get the paths of the images in a folder and store them in a DataFrame

    :param folder_path: path of the folder where the images are stored
    :param df: DataFrame where the paths will be stored
    :param n: number of images to get | default = 200

    :return: DataFrame with the paths of the images
    """
    i = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                id = len(df)
                path = os.path.join(root, file)
                df.loc[id] = [id, path]
                i += 1
                if i == n:
                    break
    return df


def ocr(path):
    """
    Function to perform OCR on an image.
    It removes non-alphanumeric characters and returns a list of words in the image.

    Args:
    path: str, path to the image
    """
    img = Image.open(path)
    text = pytesseract.image_to_string(
        img,
        lang="eng",
        config="--psm 11, -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789, --oem 3",
    )
    # text = text.replace('\n', ' ')
    # text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = text.split()
    # remove words with less than 3 characters
    text = [word for word in text if len(word) > 2]

    return text


def histogram(path, th=False, plot=False):
    """
    Function to calculate the histogram of an image

    Args:
    path: str, path to the image
    th: int, threshold to use in the histogram
    plot: bool, if True, plot the histogram

    Returns:
    list, histogram of the image
    """
    img = cv2.imread(path)

    # calculate the red channel histogram
    red_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([img], [1], None, [256], [0, 256])
    blue_hist = cv2.calcHist([img], [2], None, [256], [0, 256])

    color = ("r", "g", "b")
    histogram = []
    for i, col in enumerate(color):
        color_hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        if th:
            histogram.append(
                color_hist[0:th]
            )  # use only the first 200 values of the histogram to avoid the white color
        else:
            histogram.append(color_hist)

    if plot:
        plt.plot(histogram[0], color="r")
        plt.plot(histogram[1], color="g")
        plt.plot(histogram[2], color="b")
        plt.show()

    return histogram


def compare_histograms(hist1, hist2, print_results=False):
    """
    Function to compare the histograms of two images

    Args:
    hist1: list, histograms of the first image
    hist2: list, histograms of the second image
    print_results: bool, if True, print the results

    Returns:
    float, correlation between the two histograms
    float, intersection between the two histograms
    """
    correlation_r = cv2.compareHist(hist1[0], hist2[0], cv2.HISTCMP_CORREL)
    intersection_r = cv2.compareHist(hist1[0], hist2[0], cv2.HISTCMP_INTERSECT)
    correlation_g = cv2.compareHist(hist1[1], hist2[1], cv2.HISTCMP_CORREL)
    intersection_g = cv2.compareHist(hist1[1], hist2[1], cv2.HISTCMP_INTERSECT)
    correlation_g = cv2.compareHist(hist1[2], hist2[2], cv2.HISTCMP_CORREL)
    intersection_g = cv2.compareHist(hist1[2], hist2[2], cv2.HISTCMP_INTERSECT)
    mean_corr = (correlation_r + correlation_g + correlation_g) / 3
    mean_int = (intersection_r + intersection_g + intersection_g) / 3
    if print_results:
        print("correlation: ", mean_corr)
        print("intersection: ", mean_int)
    return mean_corr, mean_int


def compare_text(text1, text2, print_results=False):
    """
    Function to compare the words in two images

    Args:
    text1: list, words in the first image
    text2: list, words in the second image
    print_results: bool, if True, print the results

    Returns:
    int, number of words in common
    """
    # count the number of words or sets of characters in common
    count = 0
    for word in text1:
        for w in text2:
            if w in word or word in w:
                count += 1

    if len(text1) == 0 or len(text2) == 0:
        similarity_txt = 0
    else:
        similarity_txt = count / len(text1)
    return count, similarity_txt


def perform_comparation(my_img_path, cars_df, n1=0.5, n2=0.5):
    """
    Function to compare an image with the images in a DataFrame

    Args:
    my_img_path: str, path to the image to compare
    cars_df: DataFrame, DataFrame with the images to compare
    n1: float, weight of the histogram comparison
    n2: float, weight of the text comparison

    Returns:

    """
    # calculate histogram of my_img query
    my_hist = histogram(my_img_path, th=200, plot=False)
    # calculate words in my_img query
    my_text = ocr(my_img_path)

    RESULTS = []
    VALUES = []

    for i, img in cars_df.iterrows():
        img_path = img["path"]
        if img_path != my_img_path:
            # calculate histogram of the image
            hist = histogram(img_path, th=200, plot=False)
            # calculate words in the image
            text = ocr(img_path)

            # compare histograms
            _, hist_comparative = compare_histograms(my_hist, hist, print_results=False)
            # compare words
            _, text_comparative = compare_text(my_text, text, print_results=False)

            value = n1 * hist_comparative + n2 * text_comparative

            RESULTS.append([img_path, hist_comparative, text_comparative])
            VALUES.append([img_path, value])

    values_df = pd.DataFrame(VALUES, columns=["car_id", "similarity_value"])
    results_df = pd.DataFrame(
        RESULTS, columns=["car_id", "hist_correlation", "text_similarity"]
    )

    return values_df, results_df
