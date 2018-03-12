# README 


The purpose of this project is to collect the main issues on smartphones customer encounter and complain about on Internet. Our app provides an interractive dashboard to train the machine learning model, create the prediction on each comment, display various results about smartphone issues and visualize and update comments and their prediction. 

## The functionnalities of the application 

The application is a complete framework to analyse the issues of smartphones. 

First, it starts with scrapping web page to get comments about smartphone issues. For now, we scrappe only web pages in English from the website BestBuy, other webpages are to come. 

Second, it builds a data base, taking into accound the labeled data set, the unlabeled data set and the results of the scrapping. 

Third, it preprocesses the data, trains the model and predicts the issue and which type of issue on each unseen comment. 

Fourth, it creates visualisation to understand the main smartphone issues, their trends over time and the corresponding smartphone brands (Iphone or Samsung). 

Finally, as a user, you can update by hand the prediction, if you see reviews that are mislabeled by the prediction (still building this functionnality). 

## Getting started 

### Prerequisites 

You can find the packages required for the app under the file requirements.txt

### Running the app 

Open the file app.py in your favorite text editor After the imports of packages, you will see three lines of code starting with backend. 

```
backend = ReviewApp("data/test_predicted_2.db")
#backend.build_data_base(labeled="data/labeled_data.csv", unlabeled="data/data_unlabeled.csv", log_file="data/name_of_scraper.log")
#backend._build_vocab(preprocess=True)
```

The first line initializes the app. 
The second build the data base. We provide you the python file with this line as a comment, since we provide you already with the data base on the github. As we said above, this data base contains also new comments from the webpages. If you want to include new comments (or build your own data base), just uncomment this line and change the name of the data base in the first line. Beware building the database takes some time. 

You can then run the app. You will find the url of your dashboard in your console. Type this url in your browser. 


### On the dashboard 

On the dashboard, you will be able to choose the way you want to train your model. After choosing it, please click on train. Once you will on train, wait (a short time for logistic regression, longer for XGBoost). The scores of the models are going to appear. 

Then you can update the predictions by clincking on the second button. This will predict if a comment is a issue or not and if yes what type of issue it is. 

Then you will be able to visualise the results your model predicted. 

At the end, you have a table containing the issues, their date and their label (if there are no label, it means the model detected an issue but could not say which one). Should there be a wrong label, you can update the prediction by hand, which will improve the overall performance for your training model (functionnality still under construction) 

## The files of our app: 


The app.py file contains all the user interface and layout of our app. 

The data folder contains the labeled and unlabeled data and the final data base.

The review_app.py file implements the ReviewApp class which does all the data manipulation (preprocessing, unsupervised learning, training, predicting, ...). You can look at the docstrings of each individual method to see what they are doing

The review_base.py file implements the ReviewBase class which is an abstraction of the database on which is based the ReviewApp, it handles the creation, update, insertion of prediction of the database to avoid the user bothering about it (writing sql or such). For POC we used a sqlite database which slows down the writing operations a lot and should be changed to more robust system (we are thinking postgres) in the future. 

The model.py file defines the tables of the data base. 

The scrapy_scrapers folder contains all the methods to scrap data from the BestBuy webpage. 