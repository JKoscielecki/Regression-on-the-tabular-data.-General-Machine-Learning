import pandas as pd
import joblib

pipeline = joblib.load('model.pkl')
hidden_test = pd.read_csv('hidden_test.csv')
# Make predictions on the hidden test dataset using the pipeline
y_pred = pipeline.predict(hidden_test)
# Save the predictions to a CSV file
predictions = pd.DataFrame({'target': y_pred})
predictions.to_csv('hidden_test_predictions.csv', index=False)
