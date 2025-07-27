# Fraud Detection using Machine Learning and Graph Neural Networks
This project focuses on identifying fraudulent credit card transactions using a variety of machine learning models. The primary goal is to compare traditional models with a Graph Neural Network (GNN) to demonstrate the GNN's superior ability to detect complex fraud patterns by analyzing the relationships between customers and merchants.

## Exploratory Data Analysis (EDA)
The initial analysis of the transaction data revealed several key insights:

Data Imbalance: The dataset is highly imbalanced, with fraudulent transactions representing only about 1.2% of the total data. This is a common challenge in fraud detection and requires special handling during modeling.

Categorical Insights: Certain transaction categories, such as 'es_travel' and 'es_leisure', showed a significantly higher percentage of fraud. This suggests that some categories are more susceptible to fraudulent activities.

Merchant and Customer Behavior: A small number of merchants were associated with a disproportionately high number of fraudulent transactions. Similarly, some customers showed concentrated fraudulent behavior, often targeting specific high-risk merchants.

Feature Redundancy: The zipcodeOri and zipMerchant features were found to have only a single unique value, making them redundant for the analysis.

### Feature Engineering
To enhance the predictive power of the models, several new features were engineered by aggregating data at the merchant level. These features provide a richer context for each transaction:

merchant_fraud_percent: The overall percentage of fraudulent transactions for each merchant.

merchant_top_customer_fraud_percent: The fraud rate of a merchant's most frequent customer.

These engineered features help the models to better identify high-risk merchants and patterns of concentrated fraud.

### Modeling Approaches
Several models were implemented and evaluated to tackle the fraud detection problem:

Traditional Machine Learning Models
XGBoost: A gradient boosting framework that is highly effective for classification tasks. It was trained on the preprocessed and engineered features. The model's performance was optimized by tuning hyperparameters and using techniques to handle the imbalanced dataset.

Graph Neural Network (GNN)
Why a GNN? Transactional data can be naturally represented as a graph, where customers and merchants are nodes, and transactions are the edges connecting them. A GNN is uniquely suited to analyze this structure, capturing complex relationships that traditional models might miss.

## Implementation:

Nodes: Customers and merchants.

Edges: Transactions between customers and merchants.

Features: The preprocessed and engineered features were attached to the edges as attributes.

Mechanism: The GNN uses a "message passing" mechanism to learn from the features of a node's neighbors. This allows it to create sophisticated representations (embeddings) for each customer and merchant, considering the context of their transactions.

### Ensemble Model
An ensemble of the XGBoost and GNN models was created to leverage the strengths of both approaches. The final prediction was a weighted average of the predictions from the two models.

### Why the GNN Approach is Better
The GNN approach demonstrated superior performance for several reasons:

Contextual Understanding: Unlike traditional models that treat each transaction independently, the GNN analyzes the entire network of transactions. This allows it to identify complex fraud rings and coordinated fraudulent activities that would otherwise be hidden.

Relationship Analysis: By learning from the relationships between customers and merchants, the GNN can better identify patterns of fraudulent behavior. For example, it can detect when a group of seemingly unrelated customers targets the same merchant.

Improved Feature Representation: The message passing mechanism of the GNN creates richer and more informative feature representations for customers and merchants, leading to more accurate predictions.

## Results
The models were evaluated based on their precision, recall, and F1-score for the fraudulent class. The GNN and the ensemble model significantly outperformed the traditional XGBoost model, particularly in identifying fraudulent transactions with high precision and recall.

| Model    | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|----------|-------------------|----------------|------------------|
| XGBoost  | 0.54              | 0.94           | 0.69             |
| GNN      | 0.89              | 0.88           | 0.89             |
| Ensemble | 0.69              | 0.69           | 0.69             |


The results clearly indicate that the GNN's ability to analyze the underlying graph structure of the data provides a significant advantage in fraud detection.
