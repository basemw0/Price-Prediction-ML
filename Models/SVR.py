# Import necessary libraries
#Keep X11_0 remove X7_4 c = 50 e = 0.001
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_correlations(df, figsize=(12, 8), cmap="coolwarm", annot=True):
    """
    Create a heatmap showing correlations between all features in the DataFrame.
    """
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=figsize)
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap=cmap,
                annot=annot,
                fmt='.2f',
                center=0,
                vmin=-1, 
                vmax=1,
                square=True,
                linewidths=0.5)
    
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()

def print_correlation_summary(df):
    """
    Print a text summary of the strongest correlations for each feature.
    """
    corr_matrix = df.corr()
    print("Strong Correlations Summary:")
    print("-" * 50)
    
    for column in corr_matrix.columns:
        correlations = corr_matrix[column].sort_values(ascending=False)
        strong_correlations = correlations[
            (correlations.index != column) & 
            (abs(correlations) >= 0.5)
        ]
        
        if len(strong_correlations) > 0:
            print(f"\n{column}:")
            for other_col, corr_value in strong_correlations.items():
                print(f"  - {other_col}: {corr_value:.3f}")

def analyze_svr_features(X, y, kernel='rbf', plot=True):
    """
    Train SVR and analyze feature importance using multiple methods.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svr = SVR(kernel=kernel)
    svr.fit(X_scaled, y)
    
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
    
    perm_importance = permutation_importance(svr, X_scaled, y, n_repeats=10, random_state=42)
    perm_scores = pd.Series(perm_importance.importances_mean, index=feature_names)
    
    correlations = np.array([abs(np.corrcoef(X_scaled[:, i], y)[0, 1]) for i in range(X_scaled.shape[1])])
    corr_scores = pd.Series(correlations, index=feature_names)
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        perm_scores.sort_values().plot(kind='barh', ax=ax1)
        ax1.set_title('Feature Importance (Permutation)')
        ax1.set_xlabel('Mean Importance')
        ax1.grid(True)
        
        corr_scores.sort_values().plot(kind='barh', ax=ax2)
        ax2.set_title('Feature Importance (Correlation-based)')
        ax2.set_xlabel('Absolute Correlation')
        ax2.grid(True)
        plt.tight_layout()
        
    return {
        'permutation_importance': perm_scores.sort_values(ascending=False),
        'correlation_importance': corr_scores.sort_values(ascending=False)
    }

# Load the data
print("Loading data...")
data = pd.read_csv("train.csv")
data2 = pd.read_csv("test.csv")

# Preprocess the training data
print("Preprocessing data...")
data = data.drop(["X3","X5","X8","X10"], axis=1)
data2 = data2.drop(["X3","X5","X8","X10"], axis=1)

# Handle missing values
data["X9"] = data["X9"].fillna(data["X9"].mode()[0])
data2["X9"] = data2["X9"].fillna(data2["X9"].mode()[0])
# data["X4"] = data["X4"].replace(0.0,pd.NA)
# data2["X4"] = data2["X4"].replace(0.0,pd.NA)
# data["X4"] = data["X4"].fillna(data["X4"].mean())
# data2["X4"] = data2["X4"].fillna(data["X4"].mean())


XTostandard = data[["X4"]]
norm = StandardScaler().fit(XTostandard)
Xnormalized = norm.transform(XTostandard)
X_Dataframe_Norm = pd.DataFrame(Xnormalized, columns=XTostandard.columns)
for column in X_Dataframe_Norm:
    data[column] = X_Dataframe_Norm[column]

XTostandard = data2[["X4"]]
Xnormalized = norm.transform(XTostandard)
X_Dataframe_Norm = pd.DataFrame(Xnormalized, columns=XTostandard.columns)
for column in X_Dataframe_Norm:
    data2[column] = X_Dataframe_Norm[column]

le = LabelEncoder()
data["X9"] = le.fit_transform(data["X9"])
data2["X9"] = le.transform(data2["X9"])

# OneHotEncode X11
oneHotEncoder = OneHotEncoder()
X11OneHotFit = oneHotEncoder.fit(data.X11.values.reshape(-1, 1))
X11OneHot = oneHotEncoder.transform(data.X11.values.reshape(-1, 1)).toarray()
dfOneHot = pd.DataFrame(X11OneHot, columns=["X11_" + str(int(i)) for i in range(X11OneHot.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)
data = data.drop("X11", axis=1)

X11OneHot = X11OneHotFit.transform(data2.X11.values.reshape(-1, 1)).toarray()
dfOneHot = pd.DataFrame(X11OneHot, columns=["X11_" + str(int(i)) for i in range(X11OneHot.shape[1])])
data2 = pd.concat([data2, dfOneHot], axis=1)
data2 = data2.drop("X11", axis=1)



# OneHotEncode X7
X7OneHotFit = oneHotEncoder.fit(data.X7.values.reshape(-1, 1))
X7OneHot = oneHotEncoder.transform(data.X7.values.reshape(-1, 1)).toarray()
dfOneHot = pd.DataFrame(X7OneHot, columns=["X7_" + str(int(i)) for i in range(X7OneHot.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)
data = data.drop("X7", axis=1)

X7OneHot = X7OneHotFit.transform(data2.X7.values.reshape(-1, 1)).toarray()
dfOneHot = pd.DataFrame(X7OneHot, columns=["X7_" + str(int(i)) for i in range(X7OneHot.shape[1])])
data2 = pd.concat([data2, dfOneHot], axis=1)
data2 = data2.drop(["X7"], axis=1)

# Drop unnecessary columns
data = data.drop(["X2","X7_1","X7_2","X7_3","X7_6","X7_7","X7_8","X7_9","X9","X1","X11_2","X7_5","X4","X7_4"], axis=1)
data2 = data2.drop(["X2","X7_1","X7_2","X7_3","X7_6","X7_7","X7_8","X7_9","X9","X1","X11_2","X7_5","X4","X7_4"], axis=1)


# For training data
XToNormal = data[["X6"]]
norm = StandardScaler().fit(XToNormal)
Xnormalized = norm.transform(XToNormal)
X_Dataframe_Norm = pd.DataFrame(Xnormalized, columns=XToNormal.columns)
for column in X_Dataframe_Norm:
    data[column] = X_Dataframe_Norm[column]

# For test data
XToNormal2 = data2[["X6"]]
Xnormalized2 = norm.transform(XToNormal2)
X_Dataframe_Norm2 = pd.DataFrame(Xnormalized2, columns=XToNormal2.columns)
for column in X_Dataframe_Norm2:
    data2[column] = X_Dataframe_Norm2[column]

# Print correlations
print("\nFeature Correlations:")
print(data.corr())

# Split features and target
X_train = data.drop("Y", axis=1)
Y_train = data["Y"]
X_test = data2
plot_all_correlations(data)
plt.show()
# Grid Search with Cross-validation
print("\nStarting grid search...")
param_grid = {
    'C': [50, 100],
    'epsilon': [0.01,0.005, 0.05],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel':['rbf']
}

svr = SVR()
grid_search = GridSearchCV(
    estimator=svr,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, Y_train)

# Print grid search results
print("\nBest parameters:", grid_search.best_params_)
print("Best CV score (RMSE):", np.sqrt(-grid_search.best_score_))

# Make predictions using best model
predictions = grid_search.predict(X_test)

# Print prediction statistics
print("\nPredictions statistics:")
print(f"Min: {predictions.min():.3f}")
print(f"Max: {predictions.max():.3f}")
print(f"Mean: {predictions.mean():.3f}")
print(f"Std: {predictions.std():.3f}")
print(predictions)
# Create and save submission
submitions = pd.DataFrame({
    'row_id': data2.index,
    'Y': predictions
})

submitions.to_csv('sample_submission.csv', index=False)
print("\nSubmission file saved as 'code.csv'")