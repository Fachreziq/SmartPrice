
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Memuat model Decision Tree Model B
MODEL_PATH = "/content/decision_tree_model_B.joblib"
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]
defaults = bundle["defaults"]

# Nama kategori harga
CLASS_LABELS = {
    "0": "Rendah",
    "1": "Menengah Bawah",
    "2": "Menengah Atas",
    "3": "Tinggi"
}

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    prediction_code = None
    error = None
    input_values = {}

    # Nilai awal form
    for feature in features:
        input_values[feature] = defaults[feature]

    if request.method == "POST":

        try:
            values = []

            # Mengambil 20 fitur sesuai urutan model
            for feature in features:

                raw_value = request.form.get(feature, "").strip()

                if raw_value == "":
                    raise ValueError(
                        f"Fitur {feature} belum diisi."
                    )

                value = float(raw_value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"Nilai {feature} tidak valid."
                    )

                values.append(value)
                input_values[feature] = raw_value

            # Membuat data sesuai format model
            if hasattr(model, "feature_names_in_"):
                data = pd.DataFrame(
                    [values],
                    columns=features
                )
            else:
                data = np.array(values).reshape(1, -1)

            # Melakukan prediksi
            result = model.predict(data)[0]

            prediction_code = str(result)
            prediction = CLASS_LABELS.get(
                prediction_code,
                prediction_code
            )

        except ValueError as e:
            error = str(e)

        except Exception as e:
            error = "Terjadi kesalahan: " + str(e)

    return render_template(
        "index.html",
        features=features,
        defaults=defaults,
        prediction=prediction,
        prediction_code=prediction_code,
        error=error,
        input_values=input_values
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
