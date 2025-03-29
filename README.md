# Smart Supply Chain: Forecasting Demand and Classifying Delivery Performance

This project addresses two core challenges in supply chain analytics:

1. Forecasting future product demand using regression modeling
2. Classifying delivery performance into Advance, On-Time, or Late deliveries

Using historical order data, the objective is to build models that help organizations make better inventory planning decisions and improve their logistics performance.

---

## Project Objective

The primary goal of this project is to develop a data-driven system that can:

- Accurately forecast the number of items that will be ordered in the future
- Predict whether an order will be delivered early, on time, or late

Forecasting demand ensures that inventory levels match customer needs. Poor forecasting leads to either overstocking (increased holding costs and waste) or understocking (missed sales and customer dissatisfaction).

Classifying delivery performance provides insights into the reliability of shipping operations. This allows the business to take proactive steps in improving logistics, reducing delays, and enhancing the customer experience.

---

## Dataset Overview

The dataset contains the following features:

- `Order Item Quantity`: Target variable for demand forecasting
- `Order Date`: Used to derive time-based features such as month and year
- `Category Id`: Indicates the type of product ordered
- `Customer Segment`: Identifies whether the order is B2B or B2C
- `Order Country`: Origin of the order
- `Customer Country`: Destination of the shipment
- `Shipping Mode`: The delivery method used (Standard, First Class, etc.)
- `Order Item Product Price`: Price of the product
- `Days for shipping (real)`: The actual time it took to ship the item
- `Days for shipment (scheduled)`: The planned shipping time

These features were selected because they provide both temporal and behavioral signals necessary for accurate forecasting and classification.

---

## Data Preprocessing

### Handling Missing Values

Missing data was identified using null-checks. Columns with few missing values were dropped. For important features, imputation techniques like filling with the mean or median were considered.

This step is critical because most machine learning models cannot handle missing values. Cleaning the data ensures model stability and prevents errors during training.

### Date Parsing and Feature Extraction

The `Order Date` column was converted to datetime format. From this, additional features were extracted:

- Month
- Year
- Weekday
- Quarter

These features were necessary to model seasonality and temporal patterns in order volume. For instance, demand might spike in December or vary significantly between weekdays and weekends.

### Encoding Categorical Variables

Categorical variables such as `Customer Segment`, `Shipping Mode`, and `Order Country` were encoded to be used in machine learning models.

- Label Encoding was applied where categories have implicit hierarchy or when using tree-based models
- One-Hot Encoding was used when categorical values were nominal and needed to be treated independently

Encoding was essential because most machine learning models require numerical inputs.

---

## Exploratory Data Analysis (EDA)

### Time Series Analysis

Aggregate order quantities were plotted over time (monthly and yearly) to identify trends and seasonality. This analysis revealed whether demand was increasing over time and which months or quarters had the highest demand.

Understanding these patterns was important before selecting features and training models.

### Geographic Demand

Demand was grouped by `Order Country` and `Customer Country`. This identified regional demand concentration and potential areas where shipping performance might differ.

This insight could be useful for regional stocking strategies or route optimization.

### Price Sensitivity

Order quantity was plotted against product price to identify price elasticity. This analysis helped determine whether customers were responsive to price changes.

If lower prices led to higher quantities ordered, this would influence both forecasting and marketing decisions.

### Category and Segment Behavior

Demand was analyzed across product categories and customer segments. This revealed which product types had consistent or seasonal demand, and whether B2B customers ordered in larger quantities than B2C.

These insights helped tailor features to segment-specific patterns.

---

## Feature Engineering

### Time-Based Features

Time-based features were derived from the `Order Date`, including:
- Month
- Weekday
- Quarter
- Day of the Year

These were added to capture regular patterns in demand, such as end-of-quarter or holiday effects.

### Lag Features (Planned)

Lag features, such as previous month’s demand, are commonly used in time-series forecasting. They provide historical context for the model to learn from.

Although not implemented in the current version, they are planned for future enhancement to improve forecast accuracy.

### Rolling Statistics

Rolling means and standard deviations (e.g., 7-day or 30-day windows) help smooth short-term noise and detect trends or seasonality over time. These features are valuable in time-series models to capture momentum or volatility.

---

## Model Building: Demand Forecasting

### Algorithms Used

Several regression models were implemented to predict future product demand:

- Linear Regression: Baseline model for measuring linear relationships
- Random Forest Regressor: Captures non-linear patterns and handles feature interactions well
- XGBoost Regressor: Gradient boosting model that often performs well in real-world data with outliers and complex relationships

Multiple models were used to compare performance and select the most accurate one for deployment.

### Model Training

The dataset was split into training and test sets using `train_test_split`. The models were trained on the training set and evaluated on the test set to simulate unseen data.

### Evaluation Metrics

The following metrics were used to evaluate performance:

- Mean Absolute Error (MAE): Measures average prediction error
- Mean Squared Error (MSE): Penalizes larger errors more heavily
- Root Mean Squared Error (RMSE): Makes MSE interpretable in original units
- R-squared (R²): Indicates how much variance is explained by the model

These metrics were selected to ensure a balanced view of model accuracy and robustness.

---

## Feature Importance

Feature importance was analyzed using tree-based models. The most influential features included:

- Month: Showed clear seasonality in order volume
- Category Id: Different product types had different ordering patterns
- Price: Had moderate influence on demand based on price sensitivity
- Customer Segment: B2B vs B2C patterns affected quantity predictions

Understanding feature importance helped validate the model and informed feature selection.

---

## Delivery Performance Classification (Planned)

A future addition to this project will include a classification model to predict whether deliveries are:

- Late
- On-Time
- Advance

This will be based on comparing the real shipping time to the scheduled shipping time.

### Planned Features

- Days for shipping (real)
- Days for shipment (scheduled)
- Shipping Mode
- Customer and Order Countries
- Order Item Quantity
- Product Category

This model will be trained using supervised classification algorithms such as Decision Trees, Random Forest, or XGBoost Classifier.

It will help logistics teams understand the likelihood of delays and allow for proactive improvements in delivery planning.

---

## Future Enhancements

- Add lag and rolling features for advanced time-series modeling
- Implement delivery classification with accuracy, precision, recall, and F1-score metrics
- Integrate external data sources like holidays, promotions, and weather
- Build an interactive dashboard using Plotly or Streamlit
- Automate retraining with scheduled pipelines or notebooks

---

## Tech Stack

- Python
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualizations
- Scikit-learn for modeling and evaluation
- XGBoost for advanced regression modeling

---

## Author

Sai Sirisha NK   
Data Science & Machine Learning Enthusiast  
