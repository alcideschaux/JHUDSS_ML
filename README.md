# Determination of class activity of quantified self measurements using a machine learning approach

 


# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement -- a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants whom were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the manner (i.e., class) in which they did the exercise.

# Getting and cleaning the data 
The training data for this project was downloaded from [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). The data for this project came from [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har). First we load the dataset in R.


```r
# Loading dataset
data <- read.csv("pml-training.csv", na.strings = c("NA", ""))
```

We then remove all columns with NA values:


```r
data <- data[, colSums(is.na(data)) == 0]
```

Finally, we remove the first 7 columns since they include data that is not useful for the analysis.


```r
data <- data[,-c(1:7)]
```

The dataset that will be used for data analysis has the following dimensions:


```r
dim(data)
```

```
## [1] 19622    53
```

# Data analysis
We will use a machine learning approach to predict the class activity using all the variables in the dataset. For doing so, we will first create train and test sets, and then we will use classification algorithms for prediction. Finally, we will select the best algorithm based on the results of the [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) and [confussion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

For the conclusion of this project we will use the winner prediction algorithm to predict the class activity in another test set (available from [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)).

Data analysis will be carried out using R version 3.2.2 "Fire Safety" running inside RStudio version 0.99.441. The following packages will be used:


```r
library(caret)
library(rpart)
library(randomForest)
```

## Data partition
We will begin our analysis by splittin the dataset in a training set (60% of the full dataset) and a testing set (40% of the full dataset).


```r
set.seed(070891)
inTrain <- createDataPartition(y = data$classe, p = .6, list = FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]
dim(training)
```

```
## [1] 11776    53
```

We will use the training set for building the prediction models and the testing set for cross validation of the models. We will use the out-of-sample error as a measurement of the model's performance. The out-of-sample error is defined as:

*OSE = 1 - AC*

in which *OSE = out-of-sample error* and *AC = accuracy* obtained from a confusion matrix.

## Decission tree
We will begin by building a decision tree using the `rpart()` function from the [rpart](https://cran.r-project.org/web/packages/rpart/index.html) package, including all predictors. We will then cross validate the model in the testing set using `predict()` and evaluate the performance of the model using the `confusionMatrix()` function.


```r
set.seed(070891)
modelFit_ct <- rpart(classe ~ ., data = training, method = "class")
modelPred_ct <- predict(modelFit_ct, testing, type = "class")
confusionMatrix(modelPred_ct, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1967  230   23   64   25
##          B   87 1008  187  137  170
##          C   59  139 1067  176  150
##          D   84  116   91  798  106
##          E   35   25    0  111  991
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7432          
##                  95% CI : (0.7334, 0.7528)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6747          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8813   0.6640   0.7800   0.6205   0.6872
## Specificity            0.9391   0.9082   0.9191   0.9395   0.9733
## Pos Pred Value         0.8519   0.6344   0.6706   0.6678   0.8528
## Neg Pred Value         0.9521   0.9185   0.9519   0.9266   0.9325
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2507   0.1285   0.1360   0.1017   0.1263
## Detection Prevalence   0.2943   0.2025   0.2028   0.1523   0.1481
## Balanced Accuracy      0.9102   0.7861   0.8495   0.7800   0.8303
```

As seen, the performace of this model is not quite good, with an accuracy of around 74%. Thus, the estimated out-of-sample error is about 26%. Clearly, a more sofisticated approach is needed.

## Random forests
Our next approach will be to use the `randomForest()` function from the [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html) package, once again including all predictors. As before, we will cross validate the model in the testing set using `predict()` and evaluate the performance of the model using the `confusionMatrix()` function.


```r
set.seed(070891)
modelFit_rf <- randomForest(classe ~ ., data = training)
modelPred_rf <- predict(modelFit_rf, testing, type = "class")
confusionMatrix(modelPred_rf, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    4    0    0    0
##          B    0 1513    4    0    0
##          C    0    1 1360    8    5
##          D    0    0    4 1275    1
##          E    0    0    0    3 1436
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9962          
##                  95% CI : (0.9945, 0.9974)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9952          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9967   0.9942   0.9914   0.9958
## Specificity            0.9993   0.9994   0.9978   0.9992   0.9995
## Pos Pred Value         0.9982   0.9974   0.9898   0.9961   0.9979
## Neg Pred Value         1.0000   0.9992   0.9988   0.9983   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1928   0.1733   0.1625   0.1830
## Detection Prevalence   0.2850   0.1933   0.1751   0.1631   0.1834
## Balanced Accuracy      0.9996   0.9980   0.9960   0.9953   0.9977
```

As seen, by using this approach the accuracy skyrocketed to 99%. Accordingly, the estimated out-of-sample error is less than 1%. Thus, we select this model for the validation process.

# Conclussion and validation
Considering the results of the confusion matrix, we pick the last model `modelFit_rf` as the best model for predicting class activity using quantified self measurements. As the final step for this project, we will use this model to predict the class activity of a new dataset composed entirely of unseen observations, as mention before.


```r
data_new <- read.csv("pml-testing.csv")
pred_new <- predict(modelFit_rf, data_new, type = "class")
pred_new
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

When we feed these results into Coursera's auto-grader for this project (see Appendix for a description of the submission process) we obtain a perfect classification in all instances. This strengthens the evidence favoring this model as the right choice for predicting class activity based on quantified self measurements.

# Appendix
For the submission of the predictions made by our final model we used the following function (thanks to [Jeff Leek](http://jtleek.com/)) to create a text file for each prediction in the test dataset.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename,
      quote = FALSE,
      row.names = FALSE,
      col.names = FALSE)
  }
}
pml_write_files(pred_new)
```

These text files were uploaded into Coursera's auto-grader for the project.
