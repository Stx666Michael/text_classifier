# text_classifier

## Features

- A question classifier, training using the dataset with 636 questions in 10 types, testing using 52 different questions

- user-input supported

- using **K-Nearest Neighbor Algorithm** and **MLP Neural Network**

## Method

- Store the question set in one text file, the corresponding **type of each question** in another text file

- Suppose there are **_n_** different words in the question set, we create a __1 * n matrix__, each word corresponds to one element in the array, in the sequence of appearance.

- Calculate the **word frequency** of each question, if any word has a frequency **greater than zero**, set the corresponding element in the array to **one**, then set other elements to **zero**.

- Suppose there are **_m_** questions in the question set, then we got **_m_** arrays with length **_n_**, each corresponds to one question.

- Read another text file and get the **labels of each question**. Pair the labels with **questions’ corresponding array**.

- Train the model with **(array_1, label_1) to (array_n, label_n)**

- Get user-input question, convert it to a __1*n matrix__ as before, use the model to **predict** its corresponding label, which gives its type

## Validation

To be continued...

