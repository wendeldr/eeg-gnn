import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, VarianceThreshold, RFECV, SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import warnings

# Suppress ConvergenceWarning for LogisticRegression
warnings.filterwarnings("ignore", category=UserWarning)


def get_top_features(X, y, feature_names, max_features=20):
    """Filter Method: Selects top features using univariate statistical tests."""
    print("Running Filter-based selection (Univariate)...")
    # Scale X for chi2 (non-negative)
    X_minmax = MinMaxScaler().fit_transform(X)
    # Standardize for f_classif, mutual_info
    X_std = StandardScaler().fit_transform(X)
    
    # Use several methods
    k = min(max_features, X.shape[1])
    methods = {
        'f_classif': SelectKBest(f_classif, k=k).fit(X_std, y),
        'mutual_info': SelectKBest(mutual_info_classif, k=k).fit(X_std, y),
        'chi2': SelectKBest(chi2, k=k).fit(X_minmax, y)
    }
    # Voting: count how many times each feature is selected
    votes = np.zeros(X.shape[1], dtype=int)
    for m in methods.values():
        votes[m.get_support(indices=True)] += 1
    # Keep features selected by at least 2 methods
    selected = np.where(votes >= 2)[0]
    # If too few, fallback to top by votes
    if len(selected) < min(10, X.shape[1]):
        selected = np.argsort(-votes)[:min(10, X.shape[1])]
    
    final_features = [feature_names[i] for i in selected]
    print(f"  > Selected {len(final_features)} features.")
    return set(final_features)

def remove_correlated_features(df, feature_names, threshold=0.7):
    """
    Iteratively removes features with high correlation.

    For each pair of features with a correlation greater than the threshold,
    it removes the one with the higher average absolute correlation with all
    other features.
    """
    features_to_check = list(feature_names)
    features_to_remove = set()
    
    # Calculate initial full correlation matrix
    full_corr_matrix = df[features_to_check].corr().abs()
    
    while True:
        # Identify pairs with correlation above the threshold
        # considering only the features not already marked for removal
        current_features = [f for f in features_to_check if f not in features_to_remove]
        if len(current_features) < 2:
            break

        corr_matrix = full_corr_matrix.loc[current_features, current_features]
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find the pair with the highest correlation
        max_corr_val = upper.max().max()
        
        if max_corr_val < threshold:
            break  # No more pairs above the threshold

        # Get the pair
        max_corr_pair = upper.stack().idxmax()
        f1, f2 = max_corr_pair
        
        # Decide which feature to remove: the one with the higher mean correlation
        mean_corr_f1 = full_corr_matrix.loc[f1, current_features].mean()
        mean_corr_f2 = full_corr_matrix.loc[f2, current_features].mean()
        
        feature_to_drop = f1 if mean_corr_f1 > mean_corr_f2 else f2
        features_to_remove.add(feature_to_drop)

    final_features = [f for f in feature_names if f not in features_to_remove]
    print(f"Removed {len(features_to_remove)} correlated features. Kept {len(final_features)}.")
    return final_features

def remove_low_variance_features(df, threshold=0.01):
    """Removes features with low variance."""
    print("Running Low-Variance feature removal...")
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    kept_features = df.columns[selector.get_support()]
    removed_count = len(df.columns) - len(kept_features)
    print(f"  > Removed {removed_count} low-variance features. Kept {len(kept_features)}.")
    return list(kept_features)

def get_wrapper_features(X, y, feature_names):
    """Wrapper Method: Selects features using Recursive Feature Elimination with CV."""
    print("Running Wrapper-based selection (RFECV)...")
    # Use a lightweight model for speed
    estimator = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=32, verbose=1)
    # RFECV will automatically find the optimal number of features
    selector = RFECV(estimator, step=1, cv=3, scoring='accuracy', n_jobs=32, verbose=1)
    selector.fit(X, y)
    
    selected_features = [feature_names[i] for i, support in enumerate(selector.support_) if support]
    print(f"  > Selected {len(selected_features)} features via RFECV.")
    return set(selected_features)

def get_embedded_features(X, y, feature_names):
    """Embedded Method: Selects features using L1 regularization (Lasso)."""
    print("Running Embedded selection (L1/Lasso)...")
    # Standardize data for L1 regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Logistic Regression with L1 penalty. C is the inverse of regularization strength.
    estimator = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', multi_class='ovr', random_state=42)
    selector = SelectFromModel(estimator)
    selector.fit(X_scaled, y)

    selected_features = [feature_names[i] for i, support in enumerate(selector.get_support()) if support]
    print(f"  > Selected {len(selected_features)} features via L1 regularization.")
    return set(selected_features)



# csv_path = "/media/dan/Data/data/renamed_mean_data.csv"

# df = pd.read_csv(csv_path, low_memory=False)

df = pd.read_feather('/media/dan/Data/git/epd_network/renamed_mean_data.feather')  


skip = ["pid","electrode_pair","electrode_a","electrode_b",
        "soz_a","soz_b","ilae","electrode_pair_names",
        "electrode_a_name","electrode_b_name","miccai_label_a",
        "miccai_label_b","age_days","age_years","soz_bin","soz_sum","etiology"]


targets = [c for c in df.columns if c not in skip]

# fix nan, inf, and super large values
# Use .loc to avoid SettingWithCopyWarning and combine masks
inf_mask = (df[targets] > 1e200) | (df[targets] < -1e200)
df.loc[:, targets] = df[targets].replace([np.inf, -np.inf], np.nan)
df.loc[:, targets] = df[targets].mask(inf_mask, np.nan)
df[targets] = df[targets].fillna(0)


# Run selection pipeline
# 0. Initial Data Loading & Preprocessing
# (Your existing data loading and z-scoring code is here)
# Efficient vectorized z-scoring per patient for all target columns at once
df_z = df.copy()
group_means = df.groupby("pid")[targets].transform("mean")
group_stds = df.groupby("pid")[targets].transform("std", ddof=0)
# Avoid division by zero: set stds of 0 to 1 (so (x-mean)/1 = x-mean)
group_stds_replaced = group_stds.replace(0, 1)
df_z[targets] = (df[targets] - group_means) / group_stds_replaced
# replace nan with 0
df_z[targets] = df_z[targets].fillna(0)
y = df_z['soz_sum'].values

# 1. Pruning Stage
pruned_features = remove_low_variance_features(df_z[targets])
indep_features = remove_correlated_features(df_z, pruned_features, threshold=0.8)

# Prepare data for selection methods
X_final = df_z[indep_features].values

# 2. Parallel Selection Stage
filter_set = get_top_features(X_final, y, indep_features, max_features=int(len(indep_features)//2))
wrapper_set = get_wrapper_features(X_final, y, indep_features)
embedded_set = get_embedded_features(X_final, y, indep_features)

# 3. Final Ensemble Voting
print("\nConsolidating results from all methods...")
all_selections = list(filter_set) + list(wrapper_set) + list(embedded_set)
vote_counts = Counter(all_selections)

# Keep features selected by at least 2 of the 3 methods
final_ensemble_features = sorted([feature for feature, count in vote_counts.items() if count >= 2])

print(f"\n--- Final Selected Features (voted by >= 2 methods): {len(final_ensemble_features)} ---")
for f in final_ensemble_features:
    print(f)

