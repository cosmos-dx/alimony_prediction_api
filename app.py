from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# Load the dataset and train the model
data = pd.read_csv('predicted_alimony.csv')
df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['Property_Location'] = label_encoder.fit_transform(df['Property_Location'])

X = df[['Male_Age', 'Property_Area(SqFt)', 'Property_Location', 'Salary (₹)', 'Feminist_Index']]
y = df['Predicted_Alimony (₹)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)


def predict_alimony(age, area, location, salary, feminist_index):
    location_encoded = label_encoder.transform([location])[0]
    input_data = pd.DataFrame([[age, area, location_encoded, salary, feminist_index]], 
                              columns=['Male_Age', 'Property_Area(SqFt)', 'Property_Location', 'Salary (₹)', 'Feminist_Index'])
    predicted_alimony = model.predict(input_data)
    return predicted_alimony[0]


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Alimony Prediction API!'})

# Create a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        age = int(request.form['age'])
        area = int(request.form['area'])
        location = request.form['location']
        salary = int(request.form['salary'])
        feminist_index = float(request.form['feminist_index'])
        
        # Get the predicted alimony
        predicted_alimony = predict_alimony(age, area, location, salary, feminist_index)
        
        # Return the prediction
        return jsonify({'predicted_alimony': f'₹{predicted_alimony:,.2f}'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
