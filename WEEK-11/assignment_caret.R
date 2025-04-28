## Name: Satya Sudha
## Xgboost using caret
## Week 11

# Loading necessary libraries
library(xgboost)
library(caret)
library(data.table)

# Setting working directory
setwd("C:/Users/pabbi/Desktop/assignment")

# Defining dataset filenames
dataset_files <- c("data_100.csv", "data_1000.csv", "data_10000.csv", "data_100000.csv", 
                   "data_1000000.csv", "data_10000000.csv")

# Looping through dataset files and perform analysis
results <- list()  # Initialize a list to store results

for (data_file in dataset_files) {
  cat("\nWorking on dataset:", data_file, "\n")
  
  # Checking if the file exists
  if (file.exists(data_file)) {
    # Load the dataset
    data <- read.csv(data_file)
    cat("Data loaded with dimensions:", dim(data), "\n")
    
    # Preparing data: Separate predictors (X) and outcome (y)
    X <- as.matrix(data[, -ncol(data)])  # All columns except the last one (outcome)
    y <- data[, ncol(data)]  # The last column is the outcome
    
    # Splitting into training and testing sets (80-20 split)
    set.seed(42)
    train_index <- createDataPartition(y, p = 0.8, list = FALSE)
    X_train <- X[train_index, ]
    y_train <- y[train_index]
    X_test <- X[-train_index, ]
    y_test <- y[-train_index]
    
    # Creating a caret training control object for cross-validation
    train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
    
    # Training the XGBoost model using caret
    start_time <- Sys.time()
    model_caret <- train(
      x = X_train, 
      y = y_train, 
      method = "xgbTree", 
      trControl = train_control, 
      tuneLength = 5
    )
    end_time <- Sys.time()
    
    # Predicting using the trained model
    y_pred <- predict(model_caret, X_test)
    
    # Calculating accuracy
    accuracy <- sum(y_pred == y_test) / length(y_test)
    time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    # Storing the results
    results[[data_file]] <- data.frame(
      Method = "XGBoost (Caret)",
      Dataset_Size = nrow(data),
      Accuracy = accuracy,
      Time_Taken = time_taken
    )
    
    cat("Accuracy:", accuracy, "Time taken:", time_taken, "seconds\n")
  } else {
    cat("Data file", data_file, "not found!\n")
  }
}

# Converting results to a data frame for display
results_df <- do.call(rbind, results)
print(results_df)

# Saving the results to a CSV file
write.csv(results_df, "xgboost_caret_results.csv", row.names = FALSE)


###The performance of XGBoost using the caret package for training, tested on different dataset sizes, reveals the expected pattern where larger datasets result in higher accuracy, but with a corresponding increase in the time taken for model fitting. The accuracy for smaller datasets (e.g., 100 rows) is relatively lower (80%) compared to larger datasets (e.g., 10 million rows), where accuracy reaches over 99%. The results reflect the model's ability to generalize better and capture more intricate patterns as more data is available. For example, with a dataset size of 100 rows, the accuracy is at 80%, but when the dataset size increases to 1,000,000, the accuracy improves to 99.21%.

###On the other hand, the training time for the model also escalates significantly as dataset size increases. For the smallest dataset, training takes just around 0.21 seconds, but for the largest dataset, it takes 405.3 seconds. This increase in time is expected due to the increased complexity of the model fitting process as the amount of data increases.


