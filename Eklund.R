### Necessary Packages ### 
library(dplyr)
library(ggplot2)
library(randomForest)


### Reading in Datasets ###

### Comparisons Data for Clustering ###
CompData <- read.csv("~/Desktop/Projects/Eklund/CompData.csv")
ls(CompData)
CompData_cleaned <- na.omit(CompData)

# Remove non-numeric columns for calculating euclidian distance #
cluster_vars <- CompData %>%
  select(-c(Player, Season, Team, Position, X))
ls(cluster_vars)

# Standardizing numeric variables # 
cluster_scaled <- scale(cluster_vars)
cluster_scaled <- na.omit(cluster_scaled)


# Calculate Euclidean Distance #
euclidean_dist <- dist(cluster_scaled, method = "euclidean")
head(as.matrix(euclidean_dist))


# Calculate Euclidean distances from William Eklund to all other players # 
william_eklund_index <- which(CompData_cleaned$Player == "William Eklund")
euclidean_dist_to_eklund <- as.matrix(euclidean_dist)[william_eklund_index, ]

# Sort the distances to get the closest players #
closest_players <- sort(euclidean_dist_to_eklund, decreasing = FALSE)
closest_player_names_all <- CompData_cleaned$Player[as.integer(names(closest_players))]
closest_player_names_all


### Contract Data Analysis ###
ContractData <- read.csv("~/Desktop/Projects/Eklund/ContractData.csv")
head(ContractData)


# Filter closest players who also have extensions #
closest_contracts <- ContractData %>%
  filter(Player %in% closest_player_names_all)

# Overview of Data #
summary(closest_contracts$AdjAAV)
summary(closest_contracts$Length)

ggplot(closest_contracts, aes(x = AdjAAV)) + 
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
  labs(title = "Distribution of AdjAAV for Similar Players", x = "AdjAAV (in Million $)", y = "Number of Players")

ggplot(closest_contracts, aes(x = Length)) + 
  geom_bar(fill = "red", color = "black") +
  labs(title = "Distribution of Contract Length for Similar Players", x = "Length (in Years)", y = "Number of Players")

ggplot(ContractData, aes(x = Length, y = AdjAAV)) +
  geom_point(color = "blue", alpha = 0.6) +  # Scatter points with some transparency
  geom_smooth(method = "lm", color = "red", se = TRUE) +  # Add a trend line
  labs(title = "Scatterplot of Contract Length vs. Adjusted AAV",
       x = "Contract Length (Years)",
       y = "Adjusted AAV (Million $)") +
  theme_minimal()



### Merge contract data with Euclidean distances ###
contract_distances <- data.frame(Player = closest_player_names_all, Distance = closest_players) %>%
  inner_join(ContractData, by = "Player")

# Sort the dataset by Distance (ascending order) to get the closest players #
contract_distances <- contract_distances %>% 
  arrange(Distance)

# Keep only the top 15 closest players #
contract_distances_top15 <- head(contract_distances, 15)

# Inverse distance weighting (players closer to Eklund have higher weights) # 
contract_distances_top15$Weight <- 1 / (contract_distances_top15$Distance + 1e-6)  # Avoid division by zero

# Weighted prediction for AAV and Length # 
predicted_AAV_weighted <- sum(contract_distances_top15$AAV * contract_distances_top15$Weight) / sum(contract_distances_top15$Weight)
predicted_Length_weighted <- sum(contract_distances_top15$Length * contract_distances_top15$Weight) / sum(contract_distances_top15$Weight)

cat("Predicted AAV (Weighted):", predicted_AAV_weighted, "Million $\n")
cat("Predicted Length (Weighted):", predicted_Length_weighted, "Years\n")



### Combined Dataset for Predictions ###
PredictionData <- read.csv("~/Desktop/Projects/Eklund/Prediction.csv")

PredictionData_cleaned <- na.omit(PredictionData)
head(PredictionData_cleaned)

### Random Forest Model for AdjAAV ###
set.seed(254) 

PredictionData_cleaned$Position <- as.factor(PredictionData_cleaned$Position)

train_index <- sample(1:nrow(PredictionData_cleaned), 0.80 * nrow(PredictionData_cleaned))
train_data <- PredictionData_cleaned[train_index, ]
test_data <- PredictionData_cleaned[-train_index, ]


rf_model_adjAAV_year <- randomForest(AdjAAV ~ Distance + Position + Age + Height + Weight + Length, 
                                     data = train_data)

predictions_adjAAV_year <- predict(rf_model_adjAAV_year, test_data)


rmse_adjAAV_year <- sqrt(mean((predictions_adjAAV_year - test_data$AdjAAV)^2))
cat("RMSE for AdjAAV Prediction (with Year):", rmse_adjAAV_year, "\n")

importance(rf_model_adjAAV_year)




### Creating Eklund's Data Frame for Predictions ###
eklund_data2 <- data.frame(
  Player = "William Eklund",
  AAV = NA,  # We do not need this value since it's the target
  AdjAAV = NA,  # We do not need this value since it's the target
  Distance = 0,  # Eklund's distance will be 0
  Position = "LW",  # Eklund's position (adjust as needed)
  Age = 22,  # Eklund's age
  Height = 71,  # Eklund's height
  Weight = 181,  # Eklund's weight
  Length = 8
)

eklund_data2$Position <- factor(eklund_data2$Position, levels = levels(train_data$Position))


eklund_data2$Position <- as.factor(eklund_data2$Position)

predicted_adjAAV_eklund2 <- predict(rf_model_adjAAV_year, eklund_data2)

cat("Predicted AdjAAV for Eklund:", predicted_adjAAV_eklund2, "\n")




### Random Forest Model to predict Length ###
rf_model_length_prob <- randomForest(as.factor(Length) ~ Distance + Position + Age + Height + Weight,
                                     data = train_data, 
                                     importance = TRUE)


length_probs <- predict(rf_model_length_prob, eklund_data2, type = "prob")


print(length_probs)




### Predict AAV for each possible length ###
contract_lengths <- c(1,2,3,4,5,6,7,8)  # Adjust based on actual lengths in dataset
aav_predictions <- sapply(contract_lengths, function(len) {
  eklund_data$Length <- len
  predict(rf_model_adjAAV_year, eklund_data)
})

### Combine into a table ###
contract_options <- data.frame(
  Length = contract_lengths,
  Predicted_AAV = aav_predictions,
  Probability = length_probs[1, contract_lengths]  # Extract corresponding probabilities
)

print(contract_options)

