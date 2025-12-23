import numpy as np
import pandas as pd

# Import from same directory (script execution)
from anomaly_logic import detect_anomaly


# -------------------------------------------------
# Dummy inputs (shape-correct)
# -------------------------------------------------

# Fake 20-cycle sequence (20, 11)
sequence_20x11 = np.random.normal(
    loc=0.0, scale=1.0, size=(20, 11)
)

# Fake single-cycle features for Isolation Forest
if_features = pd.DataFrame(
    [np.random.normal(size=11)],
    columns=[
        "V_mean", "V_std", "V_min", "V_max", "V_range",
        "dV_dt_mean", "dV_dt_max",
        "T_mean", "T_max", "T_delta",
        "duration_s"
    ]
)


# -------------------------------------------------
# Run anomaly detection
# -------------------------------------------------

result = detect_anomaly(
    sequence_20x11=sequence_20x11,
    if_features_row=if_features
)

print("\nANOMALY RESULT")
print(result)
