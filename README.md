# HMDA_Information_retrieval

This repository is meant to be used for _Information Retrieval, Extraction and Integration_ subject of MSc. Digital Innovation: Health and Medical Data Analytics (Universidad Politécnica de Madrid).

Authors: 
* Andrés Román de Vicente Muñoz
* Laura Teresa Martínez Marquina
* Paula Manso Zorrilla

## First Assignment: Profile-based retrieval

Implementation of an information retrieval engine targeted at delivering small text snippets to different users depending on their profile. 

The [dataset](data/Movies) found in Kaggle contains the title, the plot and the gender of several movies. (User profile to be determined).

The Notebook can be found [here](Assignment1/profile_based_retrieval.ipynb).



## Second Assignment: Non-textual data extraction 
Implementation of a "toy" Content Based Information Retrieval system which recieves a Query Image and returns the 5 most similar images from the datased based on two descriptors.

The [dataset](data/Formula_one_cars) contains sevaral Formula One images from diferent teams (Alpha Tauri, Ferrari, McLaren, Mercedes, Racing Point, RedBull, Renault, Williams).

The Notebook can be found [here](Assignment2/non_textual_data_extraction.ipynb), which use some functions found [functions.py](Assignment2/functions.py). Also, the [non_textual_data_extraction.py](Assignment2/non_textual_data_extraction.py) allows the user to select a query image (introducing a number) and returns the 5 most similar images.


## Third Assignment: ML Ranking

1. Choose a type of MLR aproach.
2. Map a set of Real-world lab test to LOINC (in-class)
3. Build the appropiate training set for the given queries:
   - "Glucose in blood"
   - "Bilirubin in plasma"
   - "White blood cells count"

Optional to get over the 70% of the grade
4. Implement model using public libraries
5. Extend dataset