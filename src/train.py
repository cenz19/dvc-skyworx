import joblib
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

TRAIN_PATH = "data/train.csv"
MODEL_PATH = "models/scorecard.pkl"

TARGET = "target"

df = pd.read_csv(TRAIN_PATH)

y = df[TARGET]
X = df.drop(columns=[TARGET])

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

# define binning
binning_process = BinningProcess(
    variable_names=X.columns.tolist(),
    categorical_variables=categorical_cols
)

# model
estimator = LogisticRegression(max_iter=1000)

# scorecard
scorecard = Scorecard(
    binning_process=binning_process,
    estimator=estimator
)

scorecard.fit(X, y)

joblib.dump(scorecard, MODEL_PATH)

print("Model trained & saved")