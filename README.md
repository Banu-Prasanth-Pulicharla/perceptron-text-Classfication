# Perceptrons, and Neural Networks for Text Classification

Implement and Evaluate Perceptrons for text classification.

Implemented the perceptron algorithm (used the perceptron training rule and not the gradient descent rule). Experimented with different values of number of iterations and the learning rate. Repeated the experiment by filtering out the stop words. Compared the accuracy of perceptron implementation with that of Naive Bayes (implemented in https://github.com/Banu-Prasanth-Pulicharla/naive-bayes-text-classfication).

## How to Run?
a. Place the file `PerceptronTextClassification.py` in a directory.  
b. Use below command to run the script -   
   ```
   python perceptron_classification.py
   ```
c. Parameters for the script would be asked now. Please provide in below format -   
   ```
   <Training Set Ham Path> <Training Set Spam Path> <Test Set Ham Path> <Test Set Spam Path> <learning Rate>
   ```
   EX:-   
   ```
   D:\data_TEMP\train\ham D:\data_TEMP\train\spam D:\data_TEMP\test\ham D:\data_TEMP\test\spam 0.01
   ```
d. That's it! Output would show the accuracies for test data with and without stopwords.
   (Epochs of 1 to 20 would be considered internally)