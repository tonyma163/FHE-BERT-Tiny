import torch
import numpy as np

def precision(correct, approx):
    absolute = np.mean(np.abs(correct - approx))
    relative = absolute / np.mean(np.abs(correct))
    return 1 - relative

def load_weight(name):
    # Load each part of the weight
    weight_parts = []
    for i in range(4):
        part = np.loadtxt(f"./new-weights-sst2/{name}{i+1}.txt", delimiter=',')
        weight_parts.append(part)
    
    # Concatenate all parts
    weight = np.concatenate(weight_parts)
    
    # Load the original shape
    shape = np.loadtxt(f"./new-weights-sst2/{name}_shape.txt", delimiter=',', dtype=int)
    
    # Reshape the weight
    return weight.reshape(shape)

# Load weights for both layers
dense_weight1 = load_weight("layer0_intermediate_weight")
dense_weight2 = load_weight("layer1_intermediate_weight")

# Load original model weights
model = torch.load('./notebooks/SST-2-BERT-tiny.bin', map_location=torch.device("cpu"))
original_weight1 = model['bert.encoder.layer.0.intermediate.dense.weight'].numpy()
original_weight2 = model['bert.encoder.layer.1.intermediate.dense.weight'].numpy()

# Calculate and print accuracy
print("Layer 0 accuracy:", precision(original_weight1, dense_weight1))
print("Layer 1 accuracy:", precision(original_weight2, dense_weight2))

def compare_weights(original, loaded, name):
    print(f"\n{name}")
    print("Original shape:", original.shape)
    print("Loaded shape:", loaded.shape)
    print("Original (first 5):", original.flatten()[:5])
    print("Loaded (first 5):", loaded.flatten()[:5])
    print("Original (last 5):", original.flatten()[-5:])
    print("Loaded (last 5):", loaded.flatten()[-5:])
    print("Are they equal?", np.allclose(original, loaded))
    if not np.allclose(original, loaded):
        diff = np.abs(original - loaded)
        max_diff_index = np.unravel_index(diff.argmax(), diff.shape)
        print(f"Max difference at index {max_diff_index}:")
        print(f"Original: {original[max_diff_index]}")
        print(f"Loaded: {loaded[max_diff_index]}")
    
    print("Original stats - Mean: {:.6f}, Std: {:.6f}, Min: {:.6f}, Max: {:.6f}".format(
        np.mean(original), np.std(original), np.min(original), np.max(original)))
    print("Loaded stats - Mean: {:.6f}, Std: {:.6f}, Min: {:.6f}, Max: {:.6f}".format(
        np.mean(loaded), np.std(loaded), np.min(loaded), np.max(loaded)))

compare_weights(original_weight1, dense_weight1, "Layer 0 Intermediate Weight")
compare_weights(original_weight2, dense_weight2, "Layer 1 Intermediate Weight")