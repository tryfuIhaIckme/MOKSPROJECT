import pandas as pd
from backend_ml import PredictorAPI

api = PredictorAPI()

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
    "Space_Group": "Fm-3m",
    "MOF_id": "ABAYIO",
    "MOF_Formula": "CuH3(CO2)3", # Био-активный металл Cu
    "Total_C": 18
}

pred_h2 = api.predict(test_mof, "Hydrogen")
pred_sf6 = api.predict(test_mof, "SF6")

print(f"H2 prediction: {pred_h2:.4f}")
print(f"SF6 prediction: {pred_sf6:.4f}")
print(f"Delta: {abs(pred_h2 - pred_sf6):.4f}")

if abs(pred_h2 - pred_sf6) > 1.0:
    print("SUCCESS: Large difference between H2 and SF6 (Physical Adequacy)")
else:
    print("WARNING: Small difference between H2 and SF6")

# Проверяем био-фичи
proc = api._preprocess_input(test_mof, "Hydrogen")
print(f"Is bioactive: {proc['is_bioactive'].values[0]}")
print(f"Linker type: {proc['linker_type'].values[0]}")
print(f"Size exclusion delta (H2): {proc['size_exclusion_delta'].values[0]:.2f}")
