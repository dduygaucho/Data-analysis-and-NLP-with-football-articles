# comp20008-2021sm2a1

The repository contains the following files:
. main.py: Contains code verifying answers.
. assignment1.py: Contains 9 functions to complete 9  assigned tasks
. data/data.json: Details of recent soccer matches in the English Premier League.
. data/football: 265  news articles about soccer matches in the English Premier League.

This project aims at performing data aggregation tasks, analyzing the distribution of values for total goals (Task 4), number of articles each club is mentioned (Task 5), calculating the similarity score for each pair of clubs (Task 6) and exploring the relationship between club’s performance and its media coverage based on data.json and 265 text files about soccer matches in English Premier League.


In order to obtain the results and visualizations, libraries such as pandas, NumPy, seaborn and matplotlib are used to perform data aggregation and visualization tasks. However, in order to deal with text formats, regular expression and os library will need to be imported. In addition, while nltk package should also be loaded and imported to extract some features such as 'stopwords' to complete task8, task9 also requires the use of sklearn library (TfidfTransformer) to determine the similarity between articles.
