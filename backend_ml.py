import pandas as pd
import numpy as np
import catboost as cb
import optuna
import shap
import pickle
import os
from typing import Dict, List, Any, Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

DATASET_PATH = "FINAL_ML_READY_DATASET.csv"
MODEL_SAVE_PATH = "symmetry_predictor.cbm"
TARGET_COL = "log(D) 1bar(cm/s)"
CATEGORICAL_FEATURES = ["Symmetry", "Shape", "Pore_PointGroup", "Crystal_System", "Space_Group", "Interaction_Symmetry"]
ID_COLS = ["MOF_id", "Gas", "Formula", "Qst (kJ/mol)", "log(D0)"]

# Numerical columns as specified in TOR
NUMERICAL_FEATURES = [
    "PLD", "LCD", "ASA (m2/g)", "Porosity", "Pore Volume (cm3/g)", "vf", "sa_acc_m2g",
    "Pore_Symmetry_Order", "Gas_Symmetry_Order", "Symmetry_Order_Ratio", "Shape_Fit_Factor",
    "KineticDiameter_A", "Polarizability_1e-25_cm3", "Quadrupole_1e-40_Cm2", "Dipole_D",
    "size_ratio", "pore_fit", "polar_match", "electrostatic"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def optimize_catboost(df_path: str, n_trials: int = 10) -> Dict[str, Any]:
    """
    Optimizes CatBoost hyperparameters using Optuna with GroupShuffleSplit by MOF_id.
    """
    print(f"Loading dataset from {df_path}...")
    df = pd.read_csv(df_path)
    
    # Drop rows with missing values in target or features if any
    df = df.dropna(subset=[TARGET_COL] + ALL_FEATURES)
    
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]
    groups = df["MOF_id"]
    
    # GroupShuffleSplit to avoid data leakage (split by MOF_id)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 1500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
            "cat_features": CATEGORICAL_FEATURES,
            "task_type": "CPU" # Can change to GPU if available
        }
        
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    print(f"Starting Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial value (MAE): {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params

def train_and_save_model(df_path: str, best_params: Dict[str, Any]):
    """
    Trains the final model on 100% of the dataset and saves it.
    """
    print("Training final model on 100% data...")
    df = pd.read_csv(df_path)
    df = df.dropna(subset=[TARGET_COL] + ALL_FEATURES)
    
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]
    
    final_params = best_params.copy()
    final_params.update({
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "verbose": 100,
        "cat_features": CATEGORICAL_FEATURES
    })
    
    model = cb.CatBoostRegressor(**final_params)
    model.fit(X, y)
    
    model.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

class PredictorAPI:
    """
    Inference API for gas permeation prediction.
    """
    GAS_DATABASE = {
        "Hydrogen": {
            "Symmetry": "D∞h", "Shape": "linear", "Formula": "H2", "MolarMass_g_mol": 2.016,
            "KineticDiameter_A": 2.89, "Polarizability_1e-25_cm3": 8.18, "Quadrupole_1e-40_Cm2": 0.0,
            "Dipole_D": 0.0, "Gas_Symmetry_Order": 100
        },
        "Helium": {
            "Symmetry": "Oh", "Shape": "monoatomic", "Formula": "He", "MolarMass_g_mol": 4.003,
            "KineticDiameter_A": 2.6, "Polarizability_1e-25_cm3": 2.05, "Quadrupole_1e-40_Cm2": 0.0,
            "Dipole_D": 0.0, "Gas_Symmetry_Order": 48
        },
        "Carbon dioxide": {
            "Symmetry": "D∞h", "Shape": "linear", "Formula": "CO2", "MolarMass_g_mol": 44.01,
            "KineticDiameter_A": 3.3, "Polarizability_1e-25_cm3": 29.1, "Quadrupole_1e-40_Cm2": -13.4,
            "Dipole_D": 0.0, "Gas_Symmetry_Order": 100
        },
        "Methane": {
            "Symmetry": "Td", "Shape": "spherical", "Formula": "CH4", "MolarMass_g_mol": 16.04,
            "KineticDiameter_A": 3.8, "Polarizability_1e-25_cm3": 25.9, "Quadrupole_1e-40_Cm2": 0.0,
            "Dipole_D": 0.0, "Gas_Symmetry_Order": 24
        },
        "Nitrogen": {
            "Symmetry": "D∞h", "Shape": "linear", "Formula": "N2", "MolarMass_g_mol": 28.01,
            "KineticDiameter_A": 3.64, "Polarizability_1e-25_cm3": 17.6, "Quadrupole_1e-40_Cm2": -4.7,
            "Dipole_D": 0.0, "Gas_Symmetry_Order": 100
        },
        "Oxygen": {
            "Symmetry": "D∞h", "Shape": "linear", "Formula": "O2", "MolarMass_g_mol": 32.0,
            "KineticDiameter_A": 3.46, "Polarizability_1e-25_cm3": 15.4, "Quadrupole_1e-40_Cm2": -1.3,
            "Dipole_D": 0.0, "Gas_Symmetry_Order": 100
        },
        "SF6": {
            "Symmetry": "Oh", "Shape": "octahedral", "Formula": "SF6", "MolarMass_g_mol": 146.06,
            "KineticDiameter_A": 5.13, "Polarizability_1e-25_cm3": 65.4, "Quadrupole_1e-40_Cm2": 0.0,
            "Dipole_D": 0.0, "Gas_Symmetry_Order": 48
        }
    }

    def __init__(self, model_path=MODEL_SAVE_PATH):
        """Initialize the API and load the pre-trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
        
        self.model = cb.CatBoostRegressor()
        self.model.load_model(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.gas_db = self.GAS_DATABASE
        self.feature_names = ALL_FEATURES

    def _preprocess_input(self, mof_params: Dict[str, Any], gas_name: str) -> pd.DataFrame:
        if gas_name not in self.GAS_DATABASE:
            raise ValueError(f"Газ '{gas_name}' не найден в базе. Доступные газы: {list(self.GAS_DATABASE.keys())}")
        
        gas_info = self.GAS_DATABASE[gas_name]
        
        # Merge input MOF params with Gas data
        data = {**mof_params, **gas_info}
        
        # Calculate derived features as per TOR
        # Interaction_Symmetry = Pore_PointGroup + "_" + Gas_Symmetry (NOTE: TOR says Gas_Symmetry, but dataset has Symmetry for gas)
        data["Interaction_Symmetry"] = f"{data['Pore_PointGroup']}_{data['Symmetry']}"
        
        data["Symmetry_Order_Ratio"] = data["Pore_Symmetry_Order"] / data["Gas_Symmetry_Order"]
        
        # Shape_Fit_Factor = (PLD - KineticDiameter_A) / (abs(Pore_Symmetry_Order - Gas_Symmetry_Order) + 0.01)
        data["Shape_Fit_Factor"] = (data["PLD"] - data["KineticDiameter_A"]) / (abs(data["Pore_Symmetry_Order"] - data["Gas_Symmetry_Order"]) + 0.01)
        
        # Other cross-features (if not provided, we calculate them from base ones)
        if "size_ratio" not in data:
            data["size_ratio"] = data["KineticDiameter_A"] / data["PLD"]
        if "pore_fit" not in data:
            data["pore_fit"] = (data["LCD"] - data["KineticDiameter_A"]) / data["LCD"]
        if "polar_match" not in data:
            data["polar_match"] = data["Polarizability_1e-25_cm3"] * data["Porosity"]
        if "electrostatic" not in data:
            data["electrostatic"] = abs(data["Quadrupole_1e-40_Cm2"]) * data["vf"]
            
        # Ensure all required features are present
        input_row = pd.DataFrame([data])
        
        # Return correctly ordered columns
        return input_row[self.feature_names]

    def predict(self, mof_params: Dict[str, Any], gas_name: str) -> float:
        input_df = self._preprocess_input(mof_params, gas_name)
        prediction = self.model.predict(input_df)[0]
        return float(prediction)

    def explain_prediction(self, mof_params: dict, gas_name: str):
        """
        Generates SHAP values for a specific prediction to explain feature importance.
        Returns the shap_values object for the single prediction.
        """
        processed_df = self._preprocess_input(mof_params, gas_name)
        shap_values = self.explainer(processed_df)
        return shap_values

    def get_shap_explanation(self, mof_params: Dict[str, Any], gas_name: str) -> Dict[str, float]:
        input_df = self._preprocess_input(mof_params, gas_name)
        
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        
        shap_values = self.explainer.shap_values(input_df)
        
        # Return feature names and their corresponding SHAP values as a dict
        explanation = dict(zip(self.feature_names, shap_values[0]))
        return explanation

if __name__ == "__main__":
    # 1. Optimize (n_trials=5 for speed in test)
    best_params = optimize_catboost(DATASET_PATH, n_trials=5)
    
    # 2. Train and Save
    train_and_save_model(DATASET_PATH, best_params)
    
    # 3. Predictor API Test
    api = PredictorAPI()
    
    # Test MOF params (similar to ABAYIO)
    test_mof = {
        "PLD": 4.3,
        "LCD": 11.4,
        "ASA (m2/g)": 1334.33,
        "Porosity": 0.66554,
        "Pore Volume (cm3/g)": 0.7012,
        "vf": 0.6122,
        "sa_acc_m2g": 1271.46,
        "Pore_Symmetry_Order": 192,
        "Pore_PointGroup": "m-3m",
        "Crystal_System": "cubic",
        "Space_Group": "Fm-3m"
    }
    
    gas = "Carbon dioxide"
    pred = api.predict(test_mof, gas)
    print(f"\n[TEST] Predicted log(D) for {gas}: {pred:.4f}")
    
    # Test SHAP
    shap_vals = api.get_shap_explanation(test_mof, gas)
    print("\n[TEST] SHAP top 3 contributors:")
    sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in sorted_shap[:3]:
        print(f"  {feat}: {val:.4f}")
