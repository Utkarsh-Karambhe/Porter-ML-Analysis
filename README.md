# 🚚 Porter Delivery Time Prediction

> **A comprehensive machine learning analysis for predicting delivery times at Porter using advanced regression techniques and deep learning models.**

## 📊 Project Overview

This project develops a robust machine learning framework to predict delivery times for Porter's logistics operations. Using a dataset of **197,428 delivery records** from 2015, the analysis implements multiple algorithms including Linear Regression, Random Forest, XGBoost, and Neural Networks to achieve high-accuracy predictions.

### 🎯 Key Achievements
- **10.14 minutes** Mean Absolute Error with XGBoost
- **12.78 minutes** Root Mean Squared Error 
- **Minimal overfitting** with robust generalization

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib
```

### Installation
```bash
git clone https://github.com/Utkarsh-Karambhe/Porter-ML-Analysis
cd Porter-ML-Analysis
pip install -r requirements.txt
```

## 📁 Project Structure

```
Porter-ML-Analysis/
│
├── dataset/
│   ├── porter_data.csv           # Raw dataset
│   └── Porter_ML_Analysis.ipynb  # Main analysis notebook
│
├── .ipynb_checkpoints/
│
└── Report.pdf                    # Detailed analysis report
```

## 📈 Dataset Information

### Dataset Characteristics
- **Total Records**: 197,428 delivery transactions
- **Time Period**: 2015
- **Features**: 14 original columns + 75 engineered features
- **Target Variable**: Delivery time (actual_delivery_time - created_at)

### Key Features
| Feature | Description | Type |
|---------|-------------|------|
| `market_id` | Market identifier | Categorical |
| `created_at` | Order creation timestamp | Datetime |
| `actual_delivery_time` | Delivery completion timestamp | Datetime |
| `store_primary_category` | Store cuisine type (70+ categories) | Categorical |
| `total_items` | Number of items in order | Numerical |
| `subtotal` | Order subtotal amount | Numerical |
| `total_onshift_partners` | Available delivery partners | Numerical |
| `total_busy_partners` | Busy delivery partners | Numerical |

## 🔧 Data Preprocessing Pipeline

### 1. Missing Value Treatment
- **Critical columns**: Removed rows with missing `market_id`, `actual_delivery_time`, `order_protocol`
- **Categorical imputation**: Filled missing `store_primary_category` with 'Unknown'
- **Numerical imputation**: Used KNN imputation (k=5) for partner-related features

### 2. Outlier Detection & Removal
- Applied **IQR method** for outlier detection
- Upper bound: **88.32 minutes** (Q3 + 1.5 × IQR)
- Removed extreme outliers (>140,000 minutes ≈ 97 days)
- Final dataset: **189,708 records**

### 3. Feature Engineering
- **Temporal features**: `hour`, `day_of_week`, `is_weekend`
- **One-hot encoding**: 75 binary columns for store categories
- **Feature scaling**: StandardScaler for numerical features
- **Final feature set**: 84 features

## 🤖 Model Development

### Algorithms Evaluated
1. **Linear Regression** (Baseline)
2. **Random Forest** (100 estimators, max_depth=10)
3. **XGBoost** (100 estimators, learning_rate=0.1, max_depth=5)
4. **TensorFlow Neural Network** (3 hidden layers with dropout)

### Neural Network Architecture
```
Input Layer (84 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU + 20% Dropout)
    ↓
Hidden Layer 2 (64 neurons, ReLU + 20% Dropout)
    ↓
Hidden Layer 3 (32 neurons, ReLU)
    ↓
Output Layer (1 neuron, Regression)
```

## 📊 Model Performance

### Results After Outlier Removal

| Model | MAE (minutes) | RMSE (minutes) | Improvement |
|-------|---------------|----------------|-------------|
| **XGBoost** 🏆 | **10.14** | **12.78** | Best Overall |
| TensorFlow NN | 10.27 | 12.94 | 2nd Best |
| Random Forest | 10.36 | 13.04 | 3rd Best |
| Linear Regression | 10.64 | 13.36 | Baseline |

### Cross-Validation Results (XGBoost)
- **5-Fold CV MAE**: 10.11 minutes
- **5-Fold CV RMSE**: 12.75 minutes

### Overfitting Analysis
- **XGBoost Generalization Gap**: 0.14 minutes MAE, 0.17 minutes RMSE
- **TensorFlow Generalization Gap**: 0.14 minutes MAE, 0.16 minutes RMSE

## 💼 Business Impact

### Key Benefits
- **Enhanced Accuracy**: 10-minute average error significantly improves delivery time estimates
- **Customer Satisfaction**: More reliable delivery predictions
- **Resource Optimization**: Better planning of delivery partners and operations
- **Operational Efficiency**: Reduced customer complaints and improved trust

### Deployment Recommendations
1. **Deploy XGBoost Model** as primary prediction engine
2. **Implement real-time monitoring** for data quality checks
3. **Set up A/B testing** framework for gradual rollout
4. **Schedule regular retraining** to maintain accuracy
5. **Enhance features** with weather, traffic, and seasonal data

## 🔍 Key Findings

1. **Outlier Impact**: Removing delivery times >88.32 minutes dramatically improved model performance
2. **Feature Importance**: Temporal features and partner availability are crucial predictors
3. **Model Robustness**: Minimal overfitting ensures reliable production deployment
4. **Scalability**: Framework can handle large datasets efficiently

## 🛠️ Future Enhancements

- [ ] **Advanced Models**: Deep learning architectures, ensemble methods

## 📚 Report

- [Detailed Analysis Report](Report.pdf)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⭐ Support

If you found this project helpful, please give it a star ⭐ and share it with others!

---

**Utkarsh Karambhe**