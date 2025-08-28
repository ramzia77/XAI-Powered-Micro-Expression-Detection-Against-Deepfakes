from tensorflow.keras.models import load_model

# Load the old .h5 model
model = load_model("deepfake_detection_model.h5", compile=False)

# Save it in the new .keras format
model.save("deepfake_detection_model.keras")
print("✅ Model converted and saved as deepfake_detection_model.keras")

