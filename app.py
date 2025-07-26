from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset
df = pd.read_csv('eamcet_2024.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('\n', '_')

# Rename for consistency
df = df.rename(columns={
    'institute_name': 'institute',
    'branch_name': 'branch'
})

# Required columns
required_columns = [
    'institute', 'place', 'branch',
    'oc_boys', 'oc_girls',
    'bc_a_boys', 'bc_a_girls',
    'bc_b_boys', 'bc_b_girls',
    'bc_c_boys', 'bc_c_girls',
    'bc_d_boys', 'bc_d_girls',
    'bc_e_boys', 'bc_e_girls',
    'sc_boys', 'sc_girls',
    'st_boys', 'st_girls',
    'ews_gen_ou', 'ews_girls_ou'
]
df = df[required_columns]
places = sorted(df['place'].dropna().unique().tolist())

# Load ML model and encoders
model = joblib.load('model.pkl')
caste_encoder = joblib.load('caste_encoder.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
branch_encoder = joblib.load('branch_encoder.pkl')
place_encoder = joblib.load('place_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html', places=places)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            rank = int(request.form['rank'])
            caste = request.form['caste'].lower()
            gender = request.form['gender'].lower()
            branch = request.form['branch'].lower()
            num_colleges = int(request.form['num_colleges'])
            selected_places = request.form.getlist('selected_places')

            if caste == "ews":
                col_name = "ews_gen_ou" if gender == "male" else "ews_girls_ou"
            else:
                col_name = f"{caste}_{'boys' if gender == 'male' else 'girls'}"

            if col_name not in df.columns:
                return render_template("result.html", error=f"Invalid caste/gender: {col_name}")

            branch_col = df['branch'].astype(str).str.lower()
            filtered = df[(branch_col == branch) & (pd.to_numeric(df[col_name], errors='coerce') >= rank)]

            if selected_places:
                filtered = filtered[filtered['place'].isin(selected_places)]

            filtered = filtered.sort_values(by=col_name).head(num_colleges)

            if filtered.empty:
                return render_template("result.html", error="No eligible colleges found.")

            # Predict ML chance for each
            chances = []
            for _, row in filtered.iterrows():
                try:
                    features = [
                        rank,
                        caste_encoder.transform([caste])[0],
                        gender_encoder.transform([gender])[0],
                        branch_encoder.transform([row['branch']])[0],
                        place_encoder.transform([row['place']])[0],
                    ]
                    prob = model.predict_proba([features])[0][1]
                    chances.append(round(prob * 100, 2))
                except Exception:
                    chances.append("N/A")

            filtered['admission_chance'] = chances
            result_data = filtered[['institute', 'place', 'branch', col_name, 'admission_chance']].to_dict(orient='records')

            return render_template("result.html", tables=result_data, rank_col=col_name)

        except Exception as e:
            return render_template("result.html", error=f"Error occurred: {str(e)}")

    return render_template('index.html', places=places)

if __name__ == '__main__':
    app.run(debug=True)
