🌾 Crop Production Analysis and Prediction

📌 Project Overview

This project focuses on analyzing and predicting crop production using Python and Machine Learning techniques.

The dataset contains crop production information for the year 2024-25 across different Indian states, districts, crops, and seasons.

📊 Dataset Information

- Total Records: 7,896
- States: 31
- Districts: 723
- Crops: 12
- Seasons: Kharif, Rabi, Summer
- Features: 7

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

🔍 Exploratory Data Analysis

- Missing value analysis
- Duplicate record detection
- Statistical summary
- Crop production distribution
- Correlation heatmap
- Top 10 crops by production
- Zero area and zero production analysis

🤖 Machine Learning Models

Five regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Extra Trees Regressor
5. Gradient Boosting Regressor

🏆 Model Performance

Extra Trees achieved the best performance among the tested models.

- MAE: 0.083832
- RMSE: 0.300191
- R² Score: 0.947492

📈 Additional Analysis

- Actual vs Predicted Production
- K-Means Clustering
- Cluster-wise analysis
- Silhouette Score: 0.495

📂 Project Files

- "Crop_Production.py" – Main Python source code
- "cropproduction.csv" – Dataset
- Screenshots – Project output and visualizations

▶️ How to Run

1. Clone this repository.
2. Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

3. Keep the CSV file in the same folder as the Python file.
4. Run:

python Crop_Production.py

🎯 Objective

To explore crop production patterns and apply machine learning techniques to predict agricultural production.

👨‍💻 Author

Ujjawal Tiwari
