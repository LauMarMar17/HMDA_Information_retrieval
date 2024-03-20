# HMDA_Information_retrieval

This repository is meant to be used for _Information Retrieval, Extraction and Integration_ subject of MSc. Digital Innovation: Health and Medical Data Analytics (Universidad Politécnica de Madrid).

Authors: 
* Andrés Román de Vicente Muñoz
* Laura Teresa Martínez Marquina
* Paula Manso Zorrilla

## First Assignment: Profile-based retrieval

* Implementation of an information retrieval engine targeted at delivering small text snippets to different users depending on their profile. 

* The [dataset](data/Movies) found in Kaggle contains the title, the plot and the gender of several movies. (User profile to be determined).

* The Notebook can be found [here](Assignment1/profile_based_retrieval.ipynb).

__Statement__
The student is required to implement an information retrieval engine targeted at delivering small text snippets to different users depending on their profile. For instance, let us suppose that we have 4 different users: the first one being interested in politics and soccer, the second in music and films, the third in cars and politics and the fourth in soccer alone. An incoming document targeted at politics should be delivered to users 1 and 3, while a document on soccer should be delivered to users 1 and 4. This assignment can be done in groups of 2-3 people, although it is also possible to do it individually.

__Mandatory part (up to 7/10 points)__
Students must submit a written report of at least 10 pages explaining the method used to encode both the documents and the user profiles, together with the algorithm used to process the queries (the more efficient, the better). In addition, the student must provide an implementation of the proposed method using the Python programming language and Jupyter Notebook. Note that all the required stuff to execute the program must be provided (i.e. automatically installed) by the notebook.

__Optional part I (up to 1.5 additional points)__
The student must conduct a detailed assessment of the performance of the created IR model using the evaluation methods discussed at class. Note that the instructor already provided sample code that implements most of them. For the evaluation of your method, you can use any dataset (documents, queries and relevance judgements) downloaded from the Internet (see e.g., TREC, SMART, OSHUMED or any other downloaded from Kaggle)

__Optional part II (up to 1.5 additional points)__
Compare the performance of your method against other methods implemented by other groups. To do so, it is required that both groups use the test set (queries + relevance judgements)


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