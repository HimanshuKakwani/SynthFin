import subprocess, sys
steps=[
    [sys.executable,"-m","research.data.download_data"],
    [sys.executable,"-c","from research.data.clean_data import clean_all; clean_all()"],
    [sys.executable,"run_prediction_experiment.py"],
    [sys.executable,"run_portfolio_experiment.py"],
    [sys.executable,"run_ablation.py"],
    [sys.executable,"run_investor_profiles.py"],
]
for s in steps:
    print("\n>>>"," ".join(s))
    subprocess.run(s,check=True)
