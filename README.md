## Users Vs Bots Classification

### Project Overview

This project aims to classify **social media profiles as either real users or automated bots**. By analyzing a wide range of profile attributes and behavioral metrics (e.g., presence of profile data, posting frequency, followers count, etc.), the goal is to develop a machine learning model that can accurately distinguish between human and bot accounts. This is crucial for maintaining platform integrity, combating misinformation, and improving user experience.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Users vs. Bots Classification](https://www.kaggle.com/datasets/juice0lover/users-vs-bots-classification)
  * **Size**: 5874 entries, 60 columns (initial)
  * **Key Features**: A mix of boolean and numerical features related to profile completeness, activity, and content. Features with many missing values (e.g., `posts_count`, `avg_likes`) were handled by dropping rows with `NaN`.
  * **Approach**:
      * Data Cleaning: Dropped rows with missing values (`dropna()`). Dropped duplicates. The number of non-null entries for many columns is low, suggesting that dropping these rows results in a much smaller, but cleaner, dataset for analysis.
      * Exploratory Data Analysis: Histograms, Boxplots, and Heatmaps were used for visualization.
      * Label Encoding: Applied to all features, including categorical and numerical ones.
      * Dimensionality Reduction: Applied `PCA` with `n_components=2` to reduce the feature space, which is a significant step that simplifies the model's task but might sacrifice some information.
      * Binary Classification: The target variable 'target' indicates 'user' (1) or 'bot' (0).
      * Models Used:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree.
  * **Best Accuracy**:
      * 96.8% with SVC.
      * 96.1% with Ridge Classifier and Decision Tree Classifier.
      * 95.7% with Random Forest Classifier and Gradient Boosting Classifier.
      * The high accuracies on the reduced feature space (2 components from PCA) suggest that the bot vs. user distinction is very clear along these principal axes.

-----

### Purpose and Applications

  * **Detect and flag bot accounts** on social media platforms to improve user trust and safety.
  * Combat spam, malicious activity, and fake engagement.
  * Support research in social network analysis and online behavior.
  * Provide a tool for platform administrators to maintain a healthy and authentic online community.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Users-Vs-Bots-Classification-Using-ML.git
cd Users-Vs-Bots-Classification-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Re-evaluating the dimensionality reduction step: Instead of reducing to just 2 components with PCA, explore different numbers of components or other techniques.
  * Implementing a more robust strategy for handling missing data, such as imputation, rather than dropping a large number of rows.
  * Performing comprehensive hyperparameter tuning and cross-validation for all models.
  * Adding explainability (e.g., SHAP or LIME) to understand which original features are the most important drivers of the classification, rather than just the PCA components.
