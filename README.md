# disaster_response_nlp
### An NLP model to categorize disaster relief messages 

This project uses data provided by [Appen](https://www.appen.com/) about text messages sent during natural disaster. There are two files, messages and categories. The messages file includes the message in the original language as well as a translated version. The categories file contains information about what kind of message was sent; did it have to do with clean water, food, or getting medical help? There are 36 possible categories each message could fall into. 

The purpose of this project is to create a NLP pipeline that can process new messages and predict which category they will fall into, so they can be routed to the correct agency. 

To do so, I first created an ETL pipeline. This pipeline loads the messages and categories files, merges, cleans, and then stores them. This means any new data from future disasters can be loaded into the same pipeline for further analysis. 

Next, I built an ML pipeline to build a model that can make predictions about new messages. This pipeline loads the data, splits it into training and test data, builds a text processing pipeline, then trains and tunes a model. This model can be used to make predictions about what categories new messages will fall into. 

Finally, I built a web app with a simple interface that allows a worker to input a message and receive an answer for what categories it falls into. This allows them to quickly categorize messages and direct help where it's needed most. 


### Libraries Used: 
The libraries used for this project include:
> pandas, version 2.2.0 <br>
> scikitlearn, version 1.4 <br>
> sqlalchemy, version 2.0.28 <br>
> numpy, version 1.26.4 <br>
> nltk version 3.6.6 


