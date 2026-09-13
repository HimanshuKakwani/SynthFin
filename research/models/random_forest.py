from sklearn.ensemble import RandomForestRegressor

def build_model():
    return RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
