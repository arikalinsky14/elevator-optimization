#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def build_model(state_size, action_size):
    """Builds the network with the given input (state) size."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(state_size,), dtype=tf.float64),
        tf.keras.layers.Dense(64, activation="elu", dtype=tf.float64),
        tf.keras.layers.Dropout(0.2),  # Prevent overfitting
        tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
        tf.keras.layers.Dense(32, activation="elu", dtype=tf.float64),
        tf.keras.layers.Dense(action_size, dtype=tf.float64)
    ])
    return model

def get_first_dense_layer(model):
    """Returns the first Dense layer found in the model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            return layer
    raise ValueError("No Dense layer found in the model.")

def transfer_weights(old_model, new_model):
    """
    Transfers weights from the old model (input size 30) to the new model (input size 31).
    For the first Dense layer, the mapping is:
      - new_kernel[0:22, :] = old_kernel[0:22, :]
      - new_kernel[22, :] = old_kernel[22, :] (first capacity neuron)
      - new_kernel[23, :] = 0 (newly added neuron)
      - new_kernel[24:31, :] = old_kernel[23:30, :] (remaining neurons)
    The bias for the Dense layer is copied directly.
    The remaining layers are copied directly if their shapes match.
    """
    # Locate the first Dense layer in each model
    old_dense = get_first_dense_layer(old_model)
    new_dense = get_first_dense_layer(new_model)
    
    # Get weights from the old Dense layer
    old_weights = old_dense.get_weights()
    if len(old_weights) != 2:
        raise ValueError("Expected 2 weights (kernel and bias) in the old Dense layer, got {}".format(len(old_weights)))
    old_kernel, old_bias = old_weights  # old_kernel shape should be (30, 64)
    
    # Get weights from the new Dense layer
    new_weights = new_dense.get_weights()
    if len(new_weights) != 2:
        raise ValueError("Expected 2 weights in the new Dense layer, got {}".format(len(new_weights)))
    new_kernel, new_bias = new_weights  # new_kernel shape should be (31, 64)
    
    # Verify expected shapes
    if old_kernel.shape != (30, 64) or new_kernel.shape != (31, 64):
        raise ValueError("Unexpected kernel shapes: old_kernel {}, new_kernel {}".format(old_kernel.shape, new_kernel.shape))
    
    # Transfer weights:
    # Copy first 22 rows (unchanged neurons)
    new_kernel[0:22, :] = old_kernel[0:22, :]
    # Copy first capacity neuron
    new_kernel[22, :] = old_kernel[22, :]
    # Set row 23 (new capacity neuron) to small random values
    new_kernel[23:24, :] = tf.keras.initializers.HeNormal()(shape=(1, 64)) * 0.01  # Tiny scale
    # Copy the remaining rows
    new_kernel[24:31, :] = old_kernel[23:30, :]
    
    # Set the updated weights in the new Dense layer (bias remains the same)
    new_dense.set_weights([new_kernel, old_bias])
    
    # For the remaining layers, copy weights layer-by-layer
    old_dense_found = False
    for old_layer, new_layer in zip(old_model.layers, new_model.layers):
        if not old_dense_found:
            if isinstance(old_layer, tf.keras.layers.Dense):
                old_dense_found = True
            continue  # Skip until after the first Dense layer
        old_w = old_layer.get_weights()
        new_w = new_layer.get_weights()
        # Copy weights if both layers have weights and shapes match
        if old_w and new_w and len(old_w) == len(new_w):
            compatible = all(ow.shape == nw.shape for ow, nw in zip(old_w, new_w))
            if compatible:
                new_layer.set_weights(old_w)
            else:
                print("Skipping layer '{}' due to shape mismatch.".format(old_layer.name))

def compare_network_weights(old_model, new_model, print_elements=5):
    """
    Compares the weights of two models layer-by-layer and prints a summary.
    
    Special handling for the first Dense layer's kernel:
      - New kernel rows 0-21 should match old kernel rows 0-21
      - New kernel row 22 should match old kernel row 22 (first capacity neuron)
      - New kernel row 23 should be near zero (new capacity neuron)
      - New kernel rows 24-30 should match old kernel rows 23-29
    """
    print("\n=== Comparing Network Weights ===\n")
    
    for layer_idx, (old_layer, new_layer) in enumerate(zip(old_model.layers, new_model.layers)):
        old_weights = old_layer.get_weights()
        new_weights = new_layer.get_weights()
        
        if not old_weights and not new_weights:
            continue
        
        print("=" * 80)
        print(f"Layer {layer_idx}: '{old_layer.name}'")
        
        for weight_idx, (old_w, new_w) in enumerate(zip(old_weights, new_weights)):
            print(f"\nWeight tensor {weight_idx}:")
            print(f"  Old shape: {old_w.shape}")
            print(f"  New shape: {new_w.shape}")
            
            if old_w.shape == new_w.shape:
                diff = new_w - old_w
                print(f"  Mean diff: {np.mean(diff):.4f}, Std diff: {np.std(diff):.4f}")
                if old_w.ndim >= 2:
                    old_preview = old_w[0][:print_elements]
                    new_preview = new_w[0][:print_elements]
                else:
                    old_preview = old_w[:print_elements]
                    new_preview = new_w[:print_elements]
                print(f"  Old preview: {old_preview}")
                print(f"  New preview: {new_preview}")
            else:
                if layer_idx == 0 and weight_idx == 0:
                    old_kernel = old_w   # Expected shape: (30, 64)
                    new_kernel = new_w   # Expected shape: (31, 64)
                    
                    # Top section: rows 0-21
                    top_old = old_kernel[0:22, :]
                    top_new = new_kernel[0:22, :]
                    diff_top = top_new - top_old
                    print("  --- Top Section (rows 0-21) ---")
                    print(f"  Mean diff: {np.mean(diff_top):.4f}, Std diff: {np.std(diff_top):.4f}")
                    print(f"  Old row 0 (first {print_elements} elems): {top_old[0][:print_elements]}")
                    print(f"  New row 0 (first {print_elements} elems): {top_new[0][:print_elements]}")
                    
                    # First capacity neuron (row 22)
                    print("  --- First Capacity Neuron (row 22) ---")
                    print(f"  Old row 22 (first {print_elements} elems): {old_kernel[22][:print_elements]}")
                    print(f"  New row 22 (first {print_elements} elems): {new_kernel[22][:print_elements]}")
                    
                    # New capacity neuron (row 23)
                    print("  --- New Capacity Neuron (row 23) ---")
                    print(f"  Mean value: {np.mean(new_kernel[23]):.4f}, Std: {np.std(new_kernel[23]):.4f}")
                    print(f"  New row 23 (first {print_elements} elems): {new_kernel[23][:print_elements]}")
                    
                    # Bottom section: new rows 24-30 should equal old rows 23-29
                    bottom_old = old_kernel[23:30, :]
                    bottom_new = new_kernel[24:31, :]
                    diff_bottom = bottom_new - bottom_old
                    print("  --- Bottom Section (rows 24-30) ---")
                    print(f"  Mean diff: {np.mean(diff_bottom):.4f}, Std diff: {np.std(diff_bottom):.4f}")
                    print(f"  Old row 23 (first {print_elements} elems): {bottom_old[0][:print_elements]}")
                    print(f"  New row 24 (first {print_elements} elems): {bottom_new[0][:print_elements]}")
                    
                    # Input Neuron Mapping
                    print("\n  --- Input Neuron Mapping ---")
                    for new_idx in range(new_kernel.shape[0]):
                        if new_idx < 22:
                            mapping = f"old index {new_idx}"
                        elif new_idx == 22:
                            mapping = f"old index {new_idx} (first capacity)"
                        elif new_idx == 23:
                            mapping = "new (near-zero initialized)"
                        else:
                            mapping = f"old index {new_idx - 1}"
                        print(f"    New neuron {new_idx:2d} -> {mapping}")
                else:
                    print("  Shapes differ and no custom comparison defined for this tensor.")
        print("\n")
    
    print("=== End of Comparison ===\n")

def main():
    # Define parameters
    action_size = 3
    old_state_size = 30  # 2-capacity network input dimension
    new_state_size = 31  # 3-capacity network input dimension
    
    old_model_path = r"C:\Users\arika\OneDrive\Desktop\Python Code\savedModels\twoCapacity"
    new_model_path = r"C:\Users\arika\OneDrive\Desktop\Python Code\savedModels\threeCapacity"
    
    print("Loading 2-capacity model from:", old_model_path)
    old_model = tf.keras.models.load_model(old_model_path)
    
    print("Building 3-capacity model with input dimension:", new_state_size)
    new_model = build_model(new_state_size, action_size)
    
    print("Transferring weights...")
    transfer_weights(old_model, new_model)
    
    # Compile the new model if further training is planned
    new_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    
    print("Saving 3-capacity model to:", new_model_path)
    new_model.save(new_model_path)
    print("Weight transfer complete.")
    compare_network_weights(old_model, new_model)

if __name__ == "__main__":
    main()