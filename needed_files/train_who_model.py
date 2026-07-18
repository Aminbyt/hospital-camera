import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print("📊 Loading who_handwashing_dataset.csv...")
df = pd.read_csv("who_handwashing_dataset.csv")

# Clean any missing rows
df = df.dropna()

print(f"📁 Total training samples loaded: {len(df)}")
print("\n--- Samples per WHO Step ---")
print(df["label"].value_counts().sort_index())

# Separate 42 landmark features (X) from the WHO step label (y)
X = df.drop("label", axis=1)
y = df["label"]

# Split data: 80% to train the AI, 20% to test its accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n🚀 Training lightweight Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
)
model.fit(X_train, y_train)

# Test the model against unseen data
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n✅ Model Accuracy on Unseen Test Data: {acc:.2f}%")
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# Export the trained model
joblib.dump(model, "who_rf_model.pkl")
print(
    "\n🎉 SUCCESS! Model saved as 'who_rf_model.pkl'. Move this file to your"
    " project root directory!"
)