setwd("/Users/calvinfrederick/Desktop/university/Y1S1/DSA1101/Data")
set.seed(1101)

diabetes = read.csv("diabetes_5050.csv"); head(diabetes)
attach(diabetes)
Diabetes_binary = as.factor(Diabetes_binary)
dim(diabetes)

#DECISION TREE
install.packages("rpart")
install.packages("rpart.plot")
library(rpart.plot)
library(rpart.plot)

#NAIVE BAYES CLASSIFIER
install.packages("e1071")
library(e1071)

#ROC, AUC CURVE DIAGNOSTICS
install.packages("ROCR") 
library(ROCR)

#KNN
install.packages("class")
library(class)

###############################################################
#DATA EXPLORATION AND SUMMARISATION#

#Response variable, Diabetes_binary (Discrete, discontinuous data)
summary(Diabetes_binary) #we can see that the responses are split equally (diabetes and no diabetes)
# 0     1 
# 35346 35346
count = table(Diabetes_binary); count
barplot(count, main = paste("Respondents having Diabetes"), xlab = "0 - Absent, 1 - Present", ylab = "Number of respondents", ylim = c(0, 40000))

###############################################################

#HighBP Analysis - Categorical Variable
HighBP = as.factor(HighBP)
plot(HighBP, main = paste("Respondents having High Blood Pressure"), xlab = "0 - Absent, 1 - Present", ylab = "Number of respondents",  ylim = c(0, 40000))

#Association with Response Variable
#We use contingency table as well as odds ratio
table = table(HighBP, Diabetes_binary);table

#Join proportion by GenHlth Status
tab = prop.table(table, "HighBP"); tab #proportion by GenHlth status

# HighBP         0         1
#       0 0.7167207 0.2832793
#       1 0.3320948 0.6679052

#Odds of having diabetes
odds_of_success = tab[2,2] / 1-tab[2,1] # odds of having diabetes = success with high BP
odds_of_success2 = tab[1,2] / 1-tab[1,1] #odds of having diabetes without highBP
odds_ratio = odds_of_success/ odds_of_success; odds_ratio

# [1] -0.774754 


# Create a contingency table
table = table(HighBP, Diabetes_binary)

# Calculate the proportion of diabetes by HighBP status
tab = prop.table(table, 1); tab  # Note: the second argument is 1 for row proportions

# Calculate the odds for each HighBP group
odds_HighBP_present = tab[2, "1"] / tab[2, "0"]  # Odds of diabetes when HighBP is present
odds_HighBP_absent = tab[1, "1"] / tab[1, "0"]   # Odds of diabetes when HighBP is absent

# Calculate the odds ratio
odds_ratio <- odds_HighBP_present / odds_HighBP_absent; odds_ratio

# [1] 5.088477

###############################################################

#GenHlth Analysis #Categorical 
GenHlth = as.factor(GenHlth)
plot(GenHlth, main = paste("Health Status of Respondents"), 
     xlab = "1 = Excellent; 2 = Very Good; 3 = Good; 4 = Fair; 5 = Poor", 
     ylab = "Number of respondents", ylim = c(0, 25000))

#Association with response variable
#We use contingency table as well as odds ratio
table = table(GenHlth, Diabetes_binary);table

#Join proportion by GenHlth Status
tab = prop.table(table, "GenHlth"); tab #proportion by GenHlth status
# GenHlth        No       Yes
#       1 0.8623521 0.1376479
#       2 0.6788949 0.3211051
#       3 0.4255773 0.5744227
#       4 0.2640758 0.7359242
#       5 0.2117769 0.7882231

#Odds of having diabetes
odds_of_success = tab[,2] / tab[,1] # odds of having diabetes = success
odds_ratio = odds_of_success/ odds_of_success[1]; odds_ratio

# 1         2         3         4         5 
# 1.000000  2.963191  8.456061 17.459007 23.317698 

#ANALYSIS: Odds ratio increases with age, as the health status of respondents decreases, 
#the higher chance of having diabetes

#An odds ratio above 1 indicates a higher chance of having diabetes compared to the reference group, 
#while an odds ratio below 1 indicates a lower chance. 
#An odds ratio of 1 would indicate no difference in the chance of having diabetes compared to the reference group.

#CHI-SQUARE TEST?


###############################################################

#Age Analysis - Categorical Variable
Age = as.factor(Age)
#Using barplot
plot(Age, main = paste("Age Cateogries in Respondents"), 
     xlab = "1 = age from 18 to 24; ….; 9 = age 60 to 64; 13 = age 80 or above",
     ylab = "Number of respondents", ylim = c(0,12000)
     )
#ANALYSIS: Most of the respondents fall within Category 10, 65-70 years of age

###############################################################

#BMI Analysis - Quantitative
#Using Histogram for analysis
hist(BMI, freq = FALSE, xlab = 'total',
     y = 'Probability', ylim = c(0, 0.08))
x = seq(0, max(BMI), length.out = length(BMI))
y = dnorm(x, mean(BMI), sd(BMI))
lines(x,y, col = "red")

#ANALYSIS: We can see that the histogram is right skewed, mean is larger than the median
summary(BMI)
var(BMI)
sd(BMI)
IQR(BMI)

#Association with response variable
#Using boxplot becuase BMI is Quantitative and Response is Categorical
boxplot(BMI ~ Diabetes_binary)

#ANALYSIS: We can see that those with diabetes generally have a higher BMI than those without due to the higher median

###############################################################

#JUSTIFICATION OF USING THESE INPUT VARIABLES

#Tree-Based to determine significant input variables
DT <- rpart(Diabetes_binary ~.,
            method = "class",
            data = diabetes,
            control = rpart.control(minsplit= 7000), #10% of the observations 
            parms=list(split ='information')
)

rpart.plot(DT, type=4, extra=2, varlen=0, faclen=0, clip.right.labs=FALSE)

#ANALYSIS: WE can see that the significant input variables are HighBP, GenHlth, Age, BMI justification of why we use these

###############################################################
#FORMING CLASSIFIER MODELS#

#SETTING UP TRAINING AND TEST DATA#  #similar to tutorial 6#
n = dim(diabetes)[1]; n
train = sample(1:n, 0.8*n); #randomly sample train data (80%)
train_data = diabetes[train,] #80%
test_data = diabetes[-train,] #20%

###############################################################

#KNN#
train_x = train_data[, c("GenHlth", "HighBP", "BMI", "Age")]
test_x = test_data[, c("GenHlth", "HighBP", "BMI", "Age")]
train_y = train_data[,c("Diabetes_binary")]
test_y = test_data[,c("Diabetes_binary")]

knn_pred = knn(train_x, test_x, train_y, k = 1)
confusion_matrix = table(knn_pred , test_y)
confusion_matrix
accuracy = sum(diag(confusion_matrix))/sum(confusion_matrix); accuracy
# >> [1] 0.731452 -> 0.731

#Analysis: Accuracy is quite high for a KNN model with the k = 1, lets see if the k value is the best

##USAGE OF N-FOLDS CROSS CV TO FIND BEST K##
X=diabetes[,c("GenHlth", "HighBP", "BMI", "Age")]; head(X) # pulling out columns of explanatories/features that you want
Y=diabetes[, c("Diabetes_binary")]; head(Y) # pulling out the response column
dim(diabetes) # 1250 data points/observations
n_folds=5 #ratio 1:4 
folds_j <- sample(rep(1:n_folds, length.out = dim(diabetes)[1] )) 
table(folds_j)

#FINDING BEST K VALUE#
K = 100 
accuracy = numeric(K) # to store the average accuracy of each k.
acc = numeric(n_folds) # to store the accuracy for each iteration of n-fold CV

for (i in 1:K){
  
  for (j in 1:n_folds) {
    
    test_j <- which(folds_j == j) # get the index of the points that will be in the test set
    pred <- knn(train=X[ -test_j, ], test=X[test_j, ], cl=Y[-test_j ], k=i) 
    
    acc[j]=mean(Y[test_j] == pred) 
    # this acc[j] = sum(diag(confusion.matrix))/sum(confusion.matrix), where confusion.matrix=table(Y[test_j],pred)
  }
  
  accuracy[i] = mean(acc)
  
}

max(accuracy)
sort(accuracy)[98:100] # the three largest accuracy
index = which(accuracy == max(accuracy)) ; index # give index which is also the value of k.

plot(x=1:100, accuracy, xlab = "K")
abline(v = index, col = "red", )


##Getting the best k
index= which(acc== max(acc))
index
max(acc)

plot(x = 1:100, accuracy, xlab = "K")
abline(v = index, col = "red", )
#With a k value of 39, we find that the KNN classifier is accurate and a k-value of 39 will be used

#KNN with K-VALUE OF 39
knn_pred = knn(train_x, test_x, train_y, k = 39)
confusion_matrix = table(knn_pred , test_y)
confusion_matrix
accuracy = sum(diag(confusion_matrix))/sum(confusion_matrix); accuracy
# >> [1] 0.7384539 --> 0.738 accuracy is actually higher


###############################################################

#MULTIPLE LINEAR REGRESSION#
Multiple_LR = lm(Diabetes_binary ~ GenHlth + HighBP + BMI + Age, data = diabetes)
summary(Multiple_LR)
# Call:
#   lm(formula = Diabetes_binary ~ GenHlth + HighBP + BMI + Age, 
#      data = diabetes)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -1.59844 -0.34359  0.02996  0.34533  1.16400 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
#   (Intercept) -0.6040476  0.0088988  -67.88   <2e-16 ***
#   GenHlth      0.1211697  0.0015452   78.42   <2e-16 ***
#   HighBP       0.1920245  0.0036338   52.84   <2e-16 ***
#   BMI          0.0127993  0.0002378   53.82   <2e-16 ***
#   Age          0.0314458  0.0005992   52.48   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.4225 on 70687 degrees of freedom
# Multiple R-squared:  0.2861,	Adjusted R-squared:  0.286 
# F-statistic:  7082 on 4 and 70687 DF,  p-value: < 2.2e-16

#Goodness of Fit#
pred_model = predict(Multiple_LR, diabetes[,c("GenHlth", "HighBP", "BMI", "Age")], type = "response")
pred_MLR = prediction(pred_model, Diabetes_binary)
roc_MLR = performance(pred_MLR, measure = "tpr", x.measure = "fpr")
plot(roc_MLR, col = "green", add = TRUE) #add = TRUE means the plot the curve in the existing curve 

#AUC calculation for MLR
auc4 = performance(pred_MLR, measure = "auc")
auc4@y.values[[1]]
# [1] 0.8111536 -> 0.811

###############################################################

#DECISION TREE#
Decision_Tree <- rpart(Diabetes_binary ~ GenHlth + HighBP + BMI + Age ,
                method = "class",
                data = diabetes,
                control = rpart.control(minsplit= 7000), #10% of the observations 
                parms=list(split ='information')
)

rpart.plot(Decision_Tree, type=4, extra=2, varlen=0, faclen=0, clip.right.labs=FALSE)

#Analysis

#Goodness of fit
#ROC for Decision Trees
pred_model = predict(Decision_Tree, diabetes[, c("GenHlth", "HighBP", "BMI", "Age")], type = "prob") #without response variable
score1 = pred_model[,2]; head(score1)

pred_DT = prediction(score1, Diabetes_binary)
roc_DT = performance(pred_DT, measure = "tpr", x.measure = "fpr")
plot(roc_DT, col = "blue") #add = TRUE means the plot the curve in the existing curve 

#AUC calculation for Decision Tree
auc1 = performance(pred_DT, measure = "auc")
auc1@y.values[[1]]
# >> 0.769996 -> 0.770

#Accuracy of Decision Tree
# X = diabetes[,c("GenHlth", "HighBP", "BMI", "Age")]; head(X) # pulling out columns of explanatories/features that you want
# Y = diabetes[, c("Diabetes_binary")]; head(Y) # pulling out the response column
# dim(diabetes) # 1250 data points/observations
# n_folds=5 #ratio 1:4 
# folds_j <- sample(rep(1:n_folds, length.out = dim(diabetes)[1] )) 
# table(folds_j)
# 
# acc = numeric(n_folds)
# 
# for (j in 1:n_folds){
#   
#   test_j <- which(folds_j == j) # get the index of the points that will be in the test set
#   train_j = diabetes[-test_j, ]
# 
#   Decision_Tree <- rpart(Diabetes_binary ~ GenHlth + HighBP + BMI + Age ,
#                          method = "class",
#                          data = diabetes,
#                          control = rpart.control(minsplit= 7000), #10% of the observations 
#                          parms=list(split ='information')
#   )
#   
#   pred = predict(Decision_Tree, diabetes[,c("GenHlth", "HighBP", "BMI", "Age")], type = "class") 
#   
#   confusion_matrix = table(pred, diabetes[, c("GenHlth", "HighBP", "BMI", "Age")]);confusion_matrix
#   
#   acc[j] = sum(diag(confusion_matrix))/sum(confusion_matrix)
# 
# }
# 
# acc
# mean(acc)

library(rpart)
library(caret) # for accuracy calculation

# Set up the data
X = diabetes[,c("GenHlth", "HighBP", "BMI", "Age")]
Y = diabetes[, "Diabetes_binary"]
n_folds = 5
folds_j = sample(rep(1:n_folds, length.out = nrow(diabetes)))

# Prepare a vector to store the accuracy for each fold
acc = numeric(n_folds)

# Run the cross-validation
for (j in 1:n_folds){
  # Split the data into training and test sets
  test_indices <- which(folds_j == j)
  train_indices <- setdiff(1:nrow(diabetes), test_indices)
  train_data <- diabetes[train_indices, ]
  test_data <- diabetes[test_indices, ]
  
  # Fit the decision tree model on the training set
  Decision_Tree <- rpart(Diabetes_binary ~ GenHlth + HighBP + BMI + Age,
                         method = "class",
                         data = train_data,
                         control = rpart.control(minsplit = 10)) # Adjusted for the actual data size
  
  # Make predictions on the test set
  predictions <- predict(Decision_Tree, newdata = test_data, type = "class")
  
  # Calculate the accuracy for the fold
  acc[j] = sum(predictions == test_data$Diabetes_binary) / length(test_indices)
}

# Output the accuracy for each fold
print(acc)

# Calculate the mean accuracy over all folds
mean_acc = mean(acc)
print(mean_acc)
# [1] 0.7264612


###############################################################

#NAIVE BAYES CLASSIFIER#
Naive_Bayes = naiveBayes(Diabetes_binary ~ GenHlth + HighBP + BMI + Age, diabetes)

#GOODNESS OF FIT#
pred_model = predict(Naive_Bayes, diabetes[, c("GenHlth", "HighBP", "BMI", "Age")], type = "raw")

score2 = pred_model[,2]  #SELECT THE SECOND COLUMN, THE PROB OF SURVIVED#

pred_NB = prediction(score2, Diabetes_binary) 
roc_NB = performance(pred_NB, measure = "tpr", x.measure = "fpr")
plot(roc_NB, col = "red", add = TRUE)

#AUC Calculation for Naive Bayes
auc2 = performance(pred_NB, "auc")@y.values[[1]]; auc2
# [1] 0.8073103 --> 0.807

#Accuracy for Naive Bayes
# Now make predictions on the test set
nb_predictions <- predict(Naive_Bayes, newdata = test_data)
# Calculate accuracy
nb_accuracy <- sum(nb_predictions == test_data$Diabetes_binary) / nrow(test_data)

# Print the accuracy
print(nb_accuracy)
# [1] 0.7322818


###############################################################

# LOGISTIC REGRESSION #
Log_R = glm(Diabetes_binary ~ GenHlth + HighBP + BMI + Age, data = diabetes[,-1], family = binomial(link = "logit"))
# Call:
#   glm(formula = Diabetes_binary ~ GenHlth + HighBP + BMI + Age, 
#       family = binomial(link = "logit"), data = diabetes[, -1])
# 
# Coefficients:
#   Estimate Std. Error     z value         Pr(>|z|)    
#   (Intercept) -6.330936   0.064286  -98.48   <2e-16 ***
#   GenHlth      0.650154   0.009118   71.30   <2e-16 ***
#   HighBP       0.893401   0.019037   46.93   <2e-16 ***
#   BMI          0.079254   0.001537   51.58   <2e-16 ***
#   Age          0.185273   0.003605   51.39   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 98000  on 70691  degrees of freedom
# Residual deviance: 74669  on 70687  degrees of freedom
# AIC: 74679
# 
# Number of Fisher Scoring iterations: 4

#

#GOODNESS OF FIT#
pred_model = predict(Log_R, type = "response")
pred_LR = prediction(pred_model, Diabetes_binary)
roc_LR = performance(pred_LR, 'tpr','fpr')
plot(roc_LR, col = "black", add = TRUE)

#AUC Calculation for LR
auc3 = performance(pred_LR , "auc")
auc3@y.values[[1]]; auc3
# [1] 0.8121801 -> 0.812

###############################################################

legend("bottomright",c("MLR", "Logistic Regression", "Decision Tree", "Naive Bayes"),
       col = c("green", "black", "blue", "red"), lty = 1)

###############################################################








