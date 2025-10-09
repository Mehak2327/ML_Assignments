#question1
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA_PATH = "usa_house_prices.csv"
df = pd.read_csv(DATA_PATH)

if 'price' not in df.columns:
    raise ValueError("Change target column name to actual column that stores price.")

X = df.drop(columns=['price']).values.astype(float)
y = df['price'].values.astype(float).reshape(-1, 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

Xb = add_bias(X_scaled)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

betas = []
r2_scores = []
predictions_per_fold = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Xb), start=1):
    X_train, X_test = Xb[train_idx], Xb[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    XtX = X_train.T.dot(X_train)
    eps = 1e-8
    beta = np.linalg.pinv(XtX + eps * np.eye(XtX.shape[0])).dot(X_train.T).dot(y_train)
    y_pred = X_test.dot(beta)
    r2 = r2_score(y_test, y_pred)
    betas.append(beta)
    r2_scores.append(r2)
    predictions_per_fold.append((test_idx, y_pred))
    print(f"Fold {fold_idx}: R2 = {r2:.4f}")

best_idx = int(np.argmax(r2_scores))
best_beta = betas[best_idx]
print(f"\nBest fold index (1-based): {best_idx+1}, R2 = {r2_scores[best_idx]:.4f}")

X_train70, X_test30, y_train70, y_test30 = train_test_split(Xb, y, test_size=0.30, random_state=42, shuffle=True)
XtX_70 = X_train70.T.dot(X_train70)
beta_70 = np.linalg.pinv(XtX_70 + eps * np.eye(XtX_70.shape[0])).dot(X_train70.T).dot(y_train70)
y_pred_30 = X_test30.dot(beta_70)
r2_30 = r2_score(y_test30, y_pred_30)
print(f"R2 on 30% test (trained on 70%): {r2_30:.4f}")

y_pred_30_using_bestfold = X_test30.dot(best_beta)
r2_30_using_bestfold = r2_score(y_test30, y_pred_30_using_bestfold)
print(f"R2 on 30% test using beta from best fold: {r2_30_using_bestfold:.4f}")

np.save("betas_all_folds.npy", np.array([b.flatten() for b in betas]))
np.save("best_beta.npy", best_beta.flatten())

#question2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "usa_house_prices.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['price']).values.astype(float)
y = df['price'].values.astype(float).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xb = np.hstack([np.ones((X_scaled.shape[0],1)), X_scaled])

X_temp, X_test, y_temp, y_test = train_test_split(Xb, y, test_size=0.30, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, shuffle=True)

def gradient_descent(X, y, lr=0.01, n_iters=1000):
    m, n = X.shape
    theta = np.zeros((n,1))
    losses = []
    for i in range(n_iters):
        preds = X.dot(theta)
        error = preds - y
        grad = (2.0/m) * X.T.dot(error)
        theta = theta - lr * grad
        loss = np.mean(error**2)
        losses.append(loss)
    return theta, losses

learning_rates = [0.001, 0.01, 0.1, 1]
results = []

for lr in learning_rates:
    theta, losses = gradient_descent(X_train, y_train, lr=lr, n_iters=1000)
    y_val_pred = X_val.dot(theta)
    y_test_pred = X_test.dot(theta)
    r2_val = r2_score(y_val, y_val_pred)
    r2_test = r2_score(y_test, y_test_pred)
    results.append({
        'lr': lr,
        'theta': theta,
        'r2_val': r2_val,
        'r2_test': r2_test,
        'final_loss': losses[-1]
    })
    print(f"LR: {lr} | R2 val: {r2_val:.4f} | R2 test: {r2_test:.4f} | final MSE: {losses[-1]:.4f}")

best = max(results, key=lambda x: x['r2_val'])
print("\nBest learning rate:", best['lr'])
print("Validation R2:", best['r2_val'])
print("Test R2:", best['r2_test'])

np.save("best_theta_q2.npy", best['theta'].flatten())

#question3
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
col_names = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration","num_doors",
             "body_style", "drive_wheels", "engine_location", "wheel_base", "length", "width",
             "height", "curb_weight", "engine_type", "num_cylinders", "engine_size", "fuel_system",
             "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm", "city_mpg",
             "highway_mpg", "price"]

df = pd.read_csv(url, names=col_names, na_values='?')
df = df.dropna(subset=['price'])

numeric_cols = ["symboling", "normalized_losses", "wheel_base", "length", "width", "height",
                "curb_weight", "engine_size", "bore", "stroke", "compression_ratio", "horsepower",
                "peak_rpm", "city_mpg", "highway_mpg", "price"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

num_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
cat_cols = [c for c in df.columns if c not in numeric_cols]
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

word_to_num = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8}
def wordnum_map(x):
    try:
        return word_to_num[str(x).strip().lower()]
    except:
        try:
            return int(x)
        except:
            return np.nan

df['num_doors'] = df['num_doors'].apply(wordnum_map)
df['num_cylinders'] = df['num_cylinders'].apply(wordnum_map)
df['num_doors'] = df['num_doors'].fillna(df['num_doors'].median())
df['num_cylinders'] = df['num_cylinders'].fillna(df['num_cylinders'].median())

df = pd.get_dummies(df, columns=['body_style', 'drive_wheels'], drop_first=True)

label_cols = ['make', 'aspiration', 'engine_location', 'fuel_type']
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df['fuel_system'] = df['fuel_system'].astype(str).str.lower().apply(lambda s: 1 if 'pfi' in s else 0)
df['engine_type'] = df['engine_type'].astype(str).str.lower().apply(lambda s: 1 if 'ohc' in s else 0)

y = df['price'].values.astype(float)
X = df.drop(columns=['price']).values.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_before_pca = r2_score(y_test, y_pred)
print(f"R2 before PCA: {r2_before_pca:.4f}")

pca = PCA(n_components=0.95, svd_solver='full')
X_reduced = pca.fit_transform(X_scaled)
print("Original feature count:", X_scaled.shape[1])
print("Reduced feature count:", X_reduced.shape[1])

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.30, random_state=42)
lr2 = LinearRegression()
lr2.fit(X_train_r, y_train_r)
y_pred_r = lr2.predict(X_test_r)
r2_after_pca = r2_score(y_test_r, y_pred_r)
print(f"R2 after PCA: {r2_after_pca:.4f}")

if r2_after_pca > r2_before_pca:
    print("PCA improved performance.")
else:
    print("PCA did not improve performance.")



