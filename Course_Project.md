Project - Practical Machine Learning
--------
2020-11-10 - Cesar Benjumea


Packages
--------
    library(caret)
    library(ggplot2)
    library(gridExtra)

Overview
--------

In this report we analyze data from accelerometers on the belt, forearm,
arm, and dumbell of 6 participants. They were asked to perform barbell
lifts correctly and incorrectly, to a total of 5 different ways, while
the sensors were recording data of their physical movements. The goal of
this project is to predict the manner in which they did the exercise for
a subset of data where the target variable is unknown.

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human ’13) . Stuttgart, Germany: ACM SIGCHI, 2013. Read more:
<a href="http://groupware.les.inf.puc-rio.br/har#ixzz6dM2KSYlk" class="uri">http://groupware.les.inf.puc-rio.br/har#ixzz6dM2KSYlk</a>

Data Processing
---------------

Importing the datasets

    testing <- read.csv("pml-testing.csv")
    training <- read.csv("pml-training.csv")

Exploratory Analysis

    dim(training); dim(testing)
    table(training$classe)
    table(training$user_name); table(testing$user_name)

There are 5 different factors in the target variable “classe”, namely:
‘A’, ‘B’, ‘C’, ‘D’, ‘E’. The training dataset contains 160 variables in
total. The data is balanced for each factor of the target variable, and
for the amount of data of each of the six participants.

Data Cleaning
-------------

### Near Zero Variance Variables Across the Dataset

There are variables that have near-zero variance, which have very little
predictive power and will be removed from the data.

    # Function that removes vars with zero variance  across the dataset
    zv_vars <- function (x) {
      # names of vars with zero variance
      nsv <- nearZeroVar(x, saveMetrics = TRUE)
      drop <- row.names(nsv[nsv$nzv==TRUE,])
      
      # Function that removes the vars with zero variance the dataset
      rm_zv <- function (x, drop) {
        x[,!(names(x)) %in% drop]
      }
      
      rm_zv(x,drop)

    }
    # function returns cleaned dataframe
    training <- zv_vars(training)

### Missing Values

The variables containing missing data were removed from the data after
concluding that they were not associated with a particular “classe” or
“username” value. Also, the non-NA:NA ratio of these variables is very
small, indicating that there is very little information that can be
extracted by the predictive model.

    # All of the variables containing NAs have exactly 19216 NA values
    missingval_cols <- sapply(training, function(x) sum(is.na(x)))
    missingval_cols[missingval_cols !=0]

    # The NA values are not associated with a particular "class" or "user_name" outcome
    # The location of the NA & non-NA values in the data seems to be random 
    table(training[!is.na(training$max_roll_belt), names(training) == "classe"])

    # Only 406 entries of the variables containing NAs have numeric values. 
    # This number is small compared to the total amount of data.
    length((training[!is.na(training$max_roll_belt), names(training) == "classe"]))

    # These variables can be safely removed from the dataset
    training <- training[,!(names(training)) %in% names(missingval_cols[missingval_cols !=0])]
    testing <- testing[,!(names(testing)) %in% names(missingval_cols[missingval_cols !=0])]

Predictive Model
----------------

As each participant has an independent variation of their own physical
movements data, the strategy to predict the “classe” outcome will be
based on a model for each of the participants.

### Non Near-Zero Variance Variables Across for Each Participant

There are near-zero variance (nzv) variables that are specific to each
participant, which were not identified while removing the nzv variables
across all participants.

Each predictive model will potentially consider a different set of
variables

    # 6 users (participants) in total
    users <- unique(training$user_name)

    # Function that returns vars that are not non-zero variance variables for each user
    zv_vars_user <- function (x, users) {
      vars_user <- list()
      for (i in 1:length(users)) {
        # names of vars with zero variance
        nsv <- nearZeroVar(x[x$user_name == users[i],], saveMetrics = TRUE)
        keep <- row.names(nsv[nsv$nzv==FALSE,])
        
        vars_user[[i]] <- keep
      }
      vars_user
    }

    vars_user <- zv_vars_user(training,users)

### Trainning the model and making predictions

After identifying the relevant variables for each participant, we build
the predictive models. Cross-validation (10-fold, 3 repeats) was used to
validate the results and assess the Accuracy.

    for (i in 1:length(users)) {
      # The index variable 'X' does not contain any meaningful information, thus it is removed from the predictors list
      # It can be used later for trace-back the data
      vars <- vars_user[[i]][vars_user[[i]] != "X"]
      # Training data for each user
      train_data <- training[training$user_name == users[i],names(training) %in% vars]
      
      # Training
      #Cross-validation parameters
      control <- trainControl(method='repeatedcv', 
                              number=10, 
                              repeats=3)
      set.seed(124)
      modFit<- train(classe ~ ., data=train_data,
                     method = "rf", 
                     trControl=control)
      # Predicting
      pred <- predict(modFit, testing[testing$user_name == users[i],names(testing) %in% c(vars,"classe")])
      
      # record predictive models
      assign(paste0("modFit_",i), modFit)
      
      # record the predictions
      assign(paste0("pred_",i), pred)
     
    }

### Model Evaluation

The Cross-Validated (10 fold, repeated 3 times) accuracy for all the
constructed models was more than 99% for all 6 participants. This
accuracy is similar to that reported by *Velloso et. al.*.

    accuracy <- c(modFit_1$results[2,2], modFit_2$results[2,2], modFit_3$results[2,2],
                  modFit_4$results[2,2], modFit_5$results[2,2], modFit_6$results[2,2])
    accuracy

    ## [1] 0.9990360 1.0000000 0.9997429 0.9987746 0.9993478 1.0000000

Consequently, the error in all cases in &lt;= 0.1%. This low error can
also be observed in the corresponding confusion matrix.

    # Confusion Matrix for the predictive model of the first participant: carlitos
    confusionMatrix(modFit_1)

    ## Cross-Validated (10 fold, repeated 3 times) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 26.8  0.0  0.0  0.0  0.0
    ##          B  0.0 22.1  0.0  0.0  0.0
    ##          C  0.0  0.0 15.8  0.0  0.0
    ##          D  0.0  0.0  0.0 15.6  0.0
    ##          E  0.0  0.0  0.0  0.0 19.5
    ##                            
    ##  Accuracy (average) : 0.999

The rest of the participants have very similar confusion Matrixes

### Predictions

The accuracy and error rate of the predictions is estimated to be
slightly greater, or similar to those obtained for the training data.
Thus, the predictions of the unknown “class” values contained in the
test dataset are:

    list( "carlitos" = as.character(pred_1), 
          "pedro" = as.character(pred_2), 
          "adelmo" = as.character(pred_3), 
          "charles" = as.character(pred_4), 
          "eurico" = as.character(pred_5), 
          "jeremy" = as.character(pred_6))

    ## $carlitos
    ## [1] "A" "B" "B"
    ## 
    ## $pedro
    ## [1] "B" "A" "B"
    ## 
    ## $adelmo
    ## [1] "A"
    ## 
    ## $charles
    ## [1] "A"
    ## 
    ## $eurico
    ## [1] "A" "B" "E" "B"
    ## 
    ## $jeremy
    ## [1] "A" "B" "E" "D" "B" "C" "A" "E"

APPENDIX
--------

Session Info

    sessionInfo()

    ## R version 3.6.3 (2020-02-29)
    ## Platform: x86_64-apple-darwin15.6.0 (64-bit)
    ## Running under: macOS Catalina 10.15.7
    ## 
    ## Matrix products: default
    ## BLAS:   /Library/Frameworks/R.framework/Versions/3.6/Resources/lib/libRblas.0.dylib
    ## LAPACK: /Library/Frameworks/R.framework/Versions/3.6/Resources/lib/libRlapack.dylib
    ## 
    ## locale:
    ## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ## [1] gridExtra_2.3   caret_6.0-86    ggplot2_3.3.2   lattice_0.20-38
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.1.0     xfun_0.19            purrr_0.3.4         
    ##  [4] reshape2_1.4.4       splines_3.6.3        colorspace_1.4-1    
    ##  [7] vctrs_0.3.4          generics_0.1.0       htmltools_0.5.0     
    ## [10] stats4_3.6.3         yaml_2.2.1           survival_3.2-7      
    ## [13] prodlim_2019.11.13   rlang_0.4.8          e1071_1.7-4         
    ## [16] ModelMetrics_1.2.2.2 pillar_1.4.6         glue_1.4.2          
    ## [19] withr_2.3.0          foreach_1.5.1        lifecycle_0.2.0     
    ## [22] plyr_1.8.6           lava_1.6.8           stringr_1.4.0       
    ## [25] timeDate_3043.102    munsell_0.5.0        gtable_0.3.0        
    ## [28] recipes_0.1.14       codetools_0.2-16     evaluate_0.14       
    ## [31] knitr_1.30           class_7.3-15         Rcpp_1.0.5          
    ## [34] scales_1.1.1         ipred_0.9-9          digest_0.6.27       
    ## [37] stringi_1.5.3        dplyr_1.0.2          grid_3.6.3          
    ## [40] tools_3.6.3          magrittr_1.5         tibble_3.0.4        
    ## [43] randomForest_4.6-14  crayon_1.3.4         pkgconfig_2.0.3     
    ## [46] ellipsis_0.3.1       MASS_7.3-51.5        Matrix_1.2-18       
    ## [49] data.table_1.13.2    pROC_1.16.2          lubridate_1.7.9     
    ## [52] gower_0.2.2          rmarkdown_2.5        iterators_1.0.13    
    ## [55] R6_2.5.0             rpart_4.1-15         nnet_7.3-12         
    ## [58] nlme_3.1-144         compiler_3.6.3

<!-- ## Additional cleaning -->
<!-- The index variable 'X' does not contain any meaningful information, thus it is removed from the data -->
<!-- ```{r, cache=TRUE, results='hide', message=FALSE, warning=FALSE} -->
<!-- training <- training[,(names(training)) != "X"] -->
<!-- testing<- testing[,(names(training)) != "X"] -->
<!-- ``` -->
