# text_classifier

## Features

- A question classifier, training using the dataset with 636 questions in 10 types, testing using 52 different questions

- user-input supported

- using K-Nearest Neighbor Algorithm and MLP Neural Network

## Method

- Store the question set in one text file, the corresponding **type of each question** in another text file

- Suppose there are **_n_** different words in the question set, we create a __1 * n matrix__, each word corresponds to one element in the array, in the sequence of appearance.

- Calculate the **word frequency** of each question, if any word has a frequency **greater than zero**, set the corresponding element in the array to **one**, then set other elements to **zero**.



