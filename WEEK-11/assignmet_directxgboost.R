
#Direct Xgboost
# Loading necessary libraries
library(xgboost)
library(data.table)

# Setting working directory
setwd("C:/Users/pabbi/Desktop/assignment")

# Defining datasets
dataset_files <- c("data_100.csv", "data_1000.csv", "data_10000.csv", "data_100000.csv", 
                   "data_1000000.csv", "data_10000000.csv")

# Looping through dataset files and performing analysis
results <- list()  # Initializing a list to store results

for (data_file in dataset_files) {
  cat("\nWorking on dataset:", data_file, "\n")
  
  # Checking if the file exists
  if (file.exists(data_file)) {
    # Loading the dataset
    data <- read.csv(data_file)
    cat("Data loaded with dimensions:", dim(data), "\n")
    
    # Prepare data: Separate predictors (X) and outcome (y)
    X <- as.matrix(data[, -ncol(data)])  # All columns except the last one (outcome)
    y <- data[, ncol(data)]  # The last column is the outcome
    
    # Splitting into training and testing sets (80-20 split)
    set.seed(42)
    train_index <- createDataPartition(y, p = 0.8, list = FALSE)
    X_train <- X[train_index, ]
    y_train <- y[train_index]
    X_test <- X[-train_index, ]
    y_test <- y[-train_index]
    
    # Training the XGBoost model with 5-fold cross-validation using caret
    train_data <- xgb.DMatrix(data = X_train, label = y_train)
    test_data <- xgb.DMatrix(data = X_test, label = y_test)
    
    # Training the model using xgboost
    params <- list(objective = "binary:logistic", eval_metric = "logloss")
    start_time <- Sys.time()
    model <- xgboost(params = params, data = train_data, nrounds = 100, verbose = 0)
    end_time <- Sys.time()
    
    # Predicting using the trained model
    y_pred <- predict(model, test_data)
    y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)  # Convert to binary outcome
    
    # Calculating accuracy
    accuracy <- sum(y_pred_binary == y_test) / length(y_test)
    time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    # Storing the results
    results[[data_file]] <- data.frame(
      Method = "XGBoost (Direct)",
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
write.csv(results_df, "xgboost_results.csv", row.names = FALSE)

##As observed from the results of the XGBoost model in R, the accuracy improves as the size of the dataset increases. 
##For the smallest dataset (100 rows), the accuracy stands at 80%, while for the largest dataset (10,000,000 rows), it increases to 99.34%. This indicates that the model is able to capture more complex patterns as it has access to more data, leading to better performance. However, this improvement in accuracy comes at the cost of increased training time. The time taken to train the model grows significantly with the dataset size, starting at 0.21 seconds for 100 rows and reaching 405.3 seconds for the largest dataset. This trend highlights the trade-off between model performance and computational resources. Larger datasets require more processing power and time, but they provide the benefit of higher accuracy. This scalability makes XGBoost a strong choice for handling large datasets, particularly in situations where prediction accuracy is critical, such as in medical diagnostics or financial forecasting. However, for scenarios where time is of the essence, smaller datasets with slightly lower accuracy may be preferred. In conclusion, while the model becomes more accurate with larger datasets, the decision to use larger datasets should consider the acceptable training time and the need for real-time predictions.









