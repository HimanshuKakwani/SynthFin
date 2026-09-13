PROFILE_LAMBDA = {"conservative": 12.0, "moderate": 5.0, "aggressive": 1.5}

def score(age, dependents, capital, drawdown_tolerance, horizon_years):
    s = 0
    s += 1 if drawdown_tolerance <= 10 else 2 if drawdown_tolerance <= 30 else 3
    s += 1 if capital <= 1_200_000 else 2 if capital <= 3_600_000 else 3
    s += 3 if dependents <= 2 else 2 if dependents <= 5 else 1
    s += 3 if age <= 40 else 2 if age <= 60 else 1
    s += 1 if horizon_years <= 3 else 2 if horizon_years <= 7 else 3
    return s

def profile_lambda(profile):
    return PROFILE_LAMBDA[profile]
