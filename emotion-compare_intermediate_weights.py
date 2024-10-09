import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

def precision(correct, approx):
    absolute = np.mean(np.abs(correct - approx))
    relative = absolute / np.mean(np.abs(correct))
    return 1 - relative

def load_weight(name, folder="new-weights-emotion"):
    # Load each part of the weight
    weight_parts = []
    for i in range(4):
        part = np.loadtxt(f"./{folder}/{name}{i+1}.txt", delimiter=',')
        weight_parts.append(part)
    
    # Concatenate all parts
    weight = np.concatenate(weight_parts)
    
    if folder == "new-weights-emotion":
        # Load the original shape
        shape = np.loadtxt(f"./{folder}/{name}_shape.txt", delimiter=',', dtype=int)
    else:
        # Assume shape for weights-emotion folder
        shape = (512, 128)  # This is the expected shape for BERT-tiny intermediate layer
    
    # Reshape the weight
    return weight.reshape(shape)

# Load weights for both layers from our saved weights
our_weight1 = load_weight("layer0_intermediate_weight")
our_weight2 = load_weight("layer1_intermediate_weight")

# Load weights from the original weights-emotion folder
original_weight1 = load_weight("layer0_intermediate_weight", "weights-emotion")
original_weight2 = load_weight("layer1_intermediate_weight", "weights-emotion")

# Load model weights
model = AutoModelForSequenceClassification.from_pretrained("gokuls/BERT-tiny-emotion-intent")
model_weight1 = model.bert.encoder.layer[0].intermediate.dense.weight.detach().numpy()
model_weight2 = model.bert.encoder.layer[1].intermediate.dense.weight.detach().numpy()

def compare_weights(model_weight, our_weight, original_weight, name):
    print(f"\n{name}")
    print("Model vs Our weights accuracy:", precision(model_weight, our_weight))
    print("Model vs Original weights accuracy:", precision(model_weight, original_weight))
    print("Our vs Original weights accuracy:", precision(our_weight, original_weight))
    
    print("\nModel weights stats - Mean: {:.6f}, Std: {:.6f}, Min: {:.6f}, Max: {:.6f}".format(
        np.mean(model_weight), np.std(model_weight), np.min(model_weight), np.max(model_weight)))
    print("Our weights stats - Mean: {:.6f}, Std: {:.6f}, Min: {:.6f}, Max: {:.6f}".format(
        np.mean(our_weight), np.std(our_weight), np.min(our_weight), np.max(our_weight)))
    print("Original weights stats - Mean: {:.6f}, Std: {:.6f}, Min: {:.6f}, Max: {:.6f}".format(
        np.mean(original_weight), np.std(original_weight), np.min(original_weight), np.max(original_weight)))
    
    print("\nShapes:")
    print("Model weight shape:", model_weight.shape)
    print("Our weight shape:", our_weight.shape)
    print("Original weight shape:", original_weight.shape)
    
    print("\nFirst 5 elements:")
    print("Model:", model_weight.flatten()[:5])
    print("Ours:", our_weight.flatten()[:5])
    print("Original:", original_weight.flatten()[:5])
    
    print("\nLast 5 elements:")
    print("Model:", model_weight.flatten()[-5:])
    print("Ours:", our_weight.flatten()[-5:])
    print("Original:", original_weight.flatten()[-5:])

compare_weights(model_weight1, our_weight1, original_weight1, "Layer 0 Intermediate Weight")
compare_weights(model_weight2, our_weight2, original_weight2, "Layer 1 Intermediate Weight")