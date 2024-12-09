# EHR_UTK

Step 1: data_preprocessing1(basic data cleaning)---->UTK_med_data_cleaned.csv
data_preprocessing1(unreasonable data manipuklation)--->UTK_med_data_cleaned_with_nan.csv

step 2: data imputation
preprocesseddata_analysis ---->data_NA; data_complete; data_mice
na_fill_knn ---->data_ml
na_fill_mean---->data_mean

step 3:MLR regression

EmergencyCsection: Run MLR in python, feature importance (use this one)

EmergencyCsection-catBMI: pending

delivery_method: prediction of vag/CS

MLR_4datasets: run  MLR regression in R for the 4 datasets 



step 4:
MICE_emergency_models: analysing the mice imputed datasets


Analysis:
question 1: analysis of labor time and BMI (traditional analysis)


        
