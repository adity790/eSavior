import pandas as pd                   # For data manipulation and analysis
from datetime import datetime         # For handling date/time
import joblib

# Scikit-learn libraries for preprocessing and modeling
from sklearn.preprocessing import LabelEncoder          # To encode categorical features as numbers
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # Our ML models

data_path = "Train.csv"  # Change this path to the actual path if different
df = pd.read_csv(data_path)  # Using tab-separated values since your data is tab-delimited

print("Columns in the DataFrame:", df.columns.tolist())
print("Data types of the columns:", df.dtypes)  # Check data types of each column

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Convert purchase_date to datetime with a specific format
df['purchase_date'] = pd.to_datetime(df['purchase_date'], format='%d-%m-%Y', errors='coerce')


# Check the first few rows of the DataFrame to see the converted dates
print("First few rows of the DataFrame after converting purchase_date:")
print(df[['purchase_date']].head())

"""df.head(): Returns the first 5 rows of the DataFrame df.
df.head(n): Returns the first n rows of the DataFrame df"""

# Check the data types of the columns to confirm the conversion
print("\nData types of the DataFrame columns:")
print(df.dtypes)

current_year = datetime.now().year
df['device_age_years'] = current_year - df['purchase_date'].dt.year

df['last_repair_date'] = pd.to_datetime(df['last_repair_date'], errors='coerce')

print("\nData types of the DataFrame columns:")
print(df.dtypes)

# Calculate days since last repair
days_since_last_repair = (datetime.now() - df['last_repair_date'])
df['days_since_last_repair'] = days_since_last_repair.dt.days.fillna(-1).astype(int)

# Check the DataFrame to see the new column
print(df[['last_repair_date', 'days_since_last_repair']].head())

df['disposal_ready'] = df['disposal_ready'].apply(lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0)

# List of categorical columns (that need to be converted to numerical values)
categorical_cols = ['device_type', 'department', 'condition', 'status', 'warranty_status', 'Damage_History', 'Replacement_Type']

# Create a dictionary to keep the label encoders for each column (optional, but useful for later use)
label_encoders = {}

# Loop over each categorical column and encode them
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the encoder

"""Machine Learning Compatibility: Many machine learning algorithms (like decision trees, random forests, etc.) 
can handle categorical variables if they are encoded as integers."""

print("\nDataFrame after encoding:")
print(df)

# Optional: Print the mapping of original values to encoded values for each column
for col in categorical_cols:
    print(f"\nMapping for {col}:")
    print(dict(zip(label_encoders[col].classes_, range(len(label_encoders[col].classes_)))))

# Define the feature columns
feature_cols = ['device_type', 'department', 'condition', 'status', 'warranty_status',
                'power_rating_watt', 'previous_repairs', 'device_age_years', 'days_since_last_repair','Original_Price', 
                'Damage_History', 'Replacement_Type']

"""This line defines a list of feature column names that will be used as input for the machine learning models. 
These features are expected to influence the target variables (estimated_lifespan_years and disposal_ready)."""

X = df[feature_cols]

""" a new DataFrame X that contains only the columns specified in feature_cols. 
This DataFrame will be used as the input features for training the models."""

y_lifespan = df['estimated_lifespan_years']  # contains the values for estimated_lifespan_years
y_disposal = df['disposal_ready']  # contains the values for disposal_ready
y_price = df['Current_Price']
y_dumb = df['dumb_price']
# Train a Random Forest Regressor for estimated_lifespan_years
lifespan_model = RandomForestRegressor(n_estimators=100, random_state=42)  # use 42 as random_state as it’ll return a shuffled dataset
# Random Forest Regressor model with 100 trees (n_estimators=100) and a fixed random state for reproducibility
lifespan_model.fit(X, y_lifespan) # model learns the relationship between the features and the target.

# Train a Random Forest Classifier for disposal_ready
disposal_model = RandomForestClassifier(n_estimators=100, random_state=42)
disposal_model.fit(X, y_disposal)

model_price = RandomForestRegressor(n_estimators=100, random_state=42)
model_price.fit(X, y_price)

model_dumb = RandomForestRegressor(n_estimators=100, random_state=42)
model_dumb.fit(X, y_dumb)

joblib.dump(lifespan_model, 'lifespan_model.pkl')  
joblib.dump(disposal_model, 'disposal_model.pkl')
joblib.dump(model_price, 'price_model.pkl')
joblib.dump(model_dumb, 'dumb_model.pkl')
for col in categorical_cols:
    joblib.dump(label_encoders[col], f'label_encoder_{col}.pkl')

# save the trained models to disk using joblib. The models can be loaded later for making predictions without needing to retrain them

"""each label encoder used for encoding categorical variables. 
Each encoder is saved with a filename that corresponds to the column it encodes. 
This allows for consistent encoding of categorical variables in future datasets."""

test_data_path = "Test_0.csv"  # Path to your test data
test_df = pd.read_csv(test_data_path, sep=',')  # Adjust the separator if needed

# Preprocess the test data
test_df['purchase_date'] = pd.to_datetime(test_df['purchase_date'], format='%d-%m-%Y', errors='coerce')
test_df['last_repair_date'] = pd.to_datetime(test_df['last_repair_date'], format='%d-%m-%Y', errors='coerce')

# Create new features
test_df['device_age_years'] = current_year - test_df['purchase_date'].dt.year
test_df['days_since_last_repair'] = (datetime.now() - test_df['last_repair_date']).dt.days.fillna(-1).astype(int)

# Encode categorical variables
for col in categorical_cols:
    le = joblib.load(f'label_encoder_{col}.pkl')  # Load the encoder for each column
    test_df[col] = le.transform(test_df[col])
"""The .pkl file extension stands for "pickle," which is a Python-specific format used for serializing and deserializing Python objects"""

X_test = test_df[feature_cols]

predicted_lifespan = lifespan_model.predict(X_test)
print(predicted_lifespan)

predicted_disposal = disposal_model.predict(X_test)
print(predicted_disposal)

predicted_price = model_price.predict(X_test)
print(predicted_price)

predicted_dumb = model_dumb.predict(X_test)
print(predicted_dumb)

test_df['predicted_lifespan_years'] = predicted_lifespan
test_df['predicted_disposal_ready'] = predicted_disposal
test_df['predicted_price'] = predicted_price
test_df['predicted_dumb_price'] = predicted_dumb


output_file_path = "Predictions_0.csv"  # Specify the output file name
test_df[['device_id', 'predicted_lifespan_years', 'predicted_disposal_ready']].to_csv(output_file_path)

