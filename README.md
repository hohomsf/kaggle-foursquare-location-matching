# Kaggle Location Matching Competition
**Competition website:** https://www.kaggle.com/competitions/foursquare-location-matching


This is a Kaggle competition of entity resolution. The following description from the official site briefly explains the goal:
> In this competition, you’ll match POIs together. Using a dataset of over one-and-a-half million Places entries heavily altered to include noise, duplications, extraneous, or incorrect information, you'll produce an algorithm that predicts which Place entries represent the same point-of-interest. Each Place entry includes attributes like the name, street address, and coordinates. Successful submissions will identify matches with the greatest accuracy.

I joined the competition just a week before the closing but I still chose this because entity resolution was a new and interesting topic for me and I was excited to learn this new skill even the competition was about to end. The final score was approximately 0.757 and the model was trained with XGBoost algorithm.

## Tech Stack Highlights
* Handling real-world big data
    *  1138812 records with significant amount of missing values and noises
* EDA
    * Descriptive statistics
    * Visualization
* Data preprocessing and feature engineering
* Blocking and pair comparison with RecordLinkage
* Hyperparameter tuning with RandomizedSearchCV
* Model training and predicting with XGBoost
    * Splitting training and test datasets
    * Model evaluation with classification report and visualization

## Motivation
This is the first Kaggle competition I participated in. The objective was not to beat so many experienced data scientists and win a prize money, but to gain a hands-on experience on handling real-world big data. 

I have done some other machine learning projects, but surprisingly entity resolution / deduplication / record linkage is an area I never tried. I believe experience in this project will be a very useful weapon when handling big data with redundancy issue. I am curious about how it works and believe I can apply a lot of knowledge I have learnt.

## Data
Datasets are available [here](https://www.kaggle.com/competitions/foursquare-location-matching/data) on the competition website.

## Workflow
![flow](my_images/flow.png)<br>
Please read the notebooks to see how I went through each step.

## Codes
```location-matching-eda-model-building.ipynb```<br>
Performs EDA and model training. (Step 1 to 3 in the Workflow chart)

```location-matching-prediction.ipynb```<br>
Predicts and submits predictions. (From step 5 to submission in the Workflow chart)

```my_functions.py```<br>
Contains custom functions used in modelling and predicting.


## Challenges
* **Blocking - Accuracy vs Efficiency**<br>
    This is a problem of "Accuracy vs Efficiency". A desirable blocking should include all possible pairs. But when the data is so big, this is nearly impossible because this will drastically increase computational costs.
    
    On the other hand, we will sacrifice some real matches if we just include limited number of potential pairs. So we need to continuously optimize the blocking algorithm for better balance between accuracy and efficiency.
* **Missing values**<br>
    There are some columns with so many missing values. The "url" column even has missing values rate of 75%, and it is quite impossible for us to try to impute such kind of values. So we have to consider which features to use with consideration of the missing values.
* **Noises**<br>
    There are intentionally designed noises to simulate the real-world situation. We need to consider blocking on more than 1 column to avoid missing some of the potential pairs.
* **Multilingual datasets**<br>
    The datasets are about places in different countries so they may not be presented in English, and in worse cases, even for the same entity, different records are sometimes presented in English and sometimes in local languages. Transformation of data may be needed to tackle this problem.
* **Imbalanced samples**<br>
    Among all possible pairs, only 0.00029% really match. It is easy for the model to tend to have a negative prediction on everything if we do not handle well during model fitting.

## Next Steps
I planned to further optimize my model even after the competition ends, but unfortunately due to an issue probably on Kaggle's side, it is not possible to do any submissions now.

But still, I have concluded some potential ways for improvement.
* Rethink about blocking. While RecordLinkage is a convenient for this job, it is worth trying more algorithms such as TD-IDF, NearestNeighbors and so on to see whether accuracy or efficiency can be improved.
* Transform all multilingual text to English.
* Compare with more classifiers like Random Forest, AdaBoost, LightGBM and the like.
* Consider utilizing GPU to accelerate the training process.
* Try parallelization with PySpark DataFrames.
* Consider using clustering instead of classification.

## References
1. https://www.kaggle.com/code/nlztrk/public-0-861-pykakasi-radian-coordinates