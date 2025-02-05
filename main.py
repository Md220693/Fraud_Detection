from src.data_preprocessing import load_and_preprocess_data
from src.train_model import train_and_evaluate_models
from src.config import DATA_PATH

X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

train_and_evaluate_models(X_train, X_test, y_train, y_test)
