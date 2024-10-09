import torch
import numpy as np
import os

# Load the model
model = torch.load('./notebooks/SST-2-BERT-tiny.bin', map_location=torch.device("cpu"))

# Create a directory to save the weights
os.makedirs('new-weights-sst2', exist_ok=True)

#
def save_weight(weight, name):
    # Convert to numpy
    weight_np = weight.cpu().detach().numpy()
    
    # The weight shape should be (512, 128)
    assert weight_np.shape == (512, 128), f"Unexpected shape: {weight_np.shape}"
    
    # Transpose to get (128, 512)
    weight_np = weight_np.T
    
    # Split into 4 parts of 128x128
    for i in range(4):
        part = weight_np[:, i*128:(i+1)*128]
        np.savetxt(f"./new-weights-sst2/{name}{i+1}.txt", part.flatten(), delimiter=',')
    
    # Save the shape for reference
    np.savetxt(f"./new-weights-sst2/{name}_shape.txt", weight.shape, delimiter=',', fmt='%d')

# Save intermediate weights for both layers
def save_weight_correct_split(weight, name):
    weight_np = weight.cpu().detach().numpy()
    
    # The weight shape should be (512, 128)
    assert weight_np.shape == (512, 128), f"Unexpected shape: {weight_np.shape}"
    
    # Transpose to (128, 512)
    weight_np = weight_np.T
    
    # Split into 4 parts of 128x128
    for i in range(4):
        part = weight_np[:, i*128:(i+1)*128]
        np.savetxt(f"./new-weights-sst2/{name}{i+1}.txt", part.flatten(), delimiter=',')
    
    # Save the shape for reference
    np.savetxt(f"./new-weights-sst2/{name}_shape.txt", weight.shape, delimiter=',', fmt='%d')

# Save intermediate bias
def save_bias(bias, name):
    # Convert to numpy
    bias_np = bias.cpu().detach().numpy()
    
    # The bias shape should be (512,)
    assert bias_np.shape == (512,), f"Unexpected shape: {bias_np.shape}"
    
    # Save the bias
    np.savetxt(f"./new-weights-sst2/{name}.txt", bias_np, delimiter=',')
    
    # Save the shape for reference
    np.savetxt(f"./new-weights-sst2/{name}_shape.txt", bias.shape, delimiter=',', fmt='%d')

# Save intermediate weights for both layers
save_weight_correct_split(model['bert.encoder.layer.0.intermediate.dense.weight'], "layer0_intermediate_weight")
save_weight_correct_split(model['bert.encoder.layer.1.intermediate.dense.weight'], "layer1_intermediate_weight")

# Save intermediate bias for both layers
save_bias(model['bert.encoder.layer.0.intermediate.dense.bias'], "layer0_intermediate_bias")
save_bias(model['bert.encoder.layer.1.intermediate.dense.bias'], "layer1_intermediate_bias")

weight_shape = model['bert.encoder.layer.0.intermediate.dense.weight'].shape
print(f"Layer 0 Intermediate Dense Weight Shape: {weight_shape}")

bias_shape = model['bert.encoder.layer.0.intermediate.dense.bias'].shape
print(f"Layer 0 Intermediate Dense Bias Shape: {bias_shape}")

print("Weights saved successfully.")