import numpy as np
import json

YOUR_MODEL_NAME = 'fashion_mnist' # Default extension is h5
TF_MODEL_PATH = f'{YOUR_MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_NAME}.json'

model = tf.keras.models.load_model(TF_MODEL_PATH)

# Save weights to .npz (NumPy format)
weights = model.get_weights()
np.savez('model_weights.npz', *weights)

# Save architecture to JSON
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model.to_json())

import tensorflow as tf
import numpy as np

# === Step 1: Load Keras .h5 model ===
model = tf.keras.models.load_model(TF_MODEL_PATH)

# === Step 2: Print and collect weights ===
params = {}
print("🔍 Extracting weights from model...\n")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        for i, w in enumerate(weights):
            param_name = f"{layer.name}_{i}"
            print(f"  {param_name}: shape={w.shape}")
            params[param_name] = w
        print()

# === Step 3: Save to .npz ===
np.savez(MODEL_WEIGHTS_PATH, **params)
print(f"✅ Saved all weights to {MODEL_WEIGHTS_PATH}")

# === Step 4: Reload and verify ===
print("\n🔁 Verifying loaded .npz weights...\n")
loaded = np.load(MODEL_WEIGHTS_PATH)

for key in loaded.files:
    print(f"{key}: shape={loaded[key].shape}")

# === Step 6: Extract architecture to JSON ===
arch = []
for layer in model.layers:
    config = layer.get_config()
    info = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": config,
        "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
    }
    arch.append(info)

with open(MODEL_ARCH_PATH, "w") as f:
    json.dump(arch, f, indent=2)

print("✅ Architecture saved to model_architecture.json")