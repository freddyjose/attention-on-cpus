import numpy as np
import ctypes
import torch
import time
import re

from attention_comfy import attention_pytorch  # From ComfyUI submodule or copied file

# gets the shape of q, k and v from shape.txt
def parse_shapes(file_path):
    shapes = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"(\w+) shape: torch\.Size\((\[.*?\])\)", line.strip())
            if match:
                key = match.group(1)
                shape_str = match.group(2)
                shape = tuple(int(x) for x in shape_str.strip('[]').split(', '))
                shapes[key] = shape
    return shapes

# loads the q,k and v vectors from stored files, just like how optimized_attention is called from layers.py file
def load_matrices(shapes, q_file, k_file, v_file):
    # Load FP32 arrays
    q = np.load(q_file)
    k = np.load(k_file)
    v = np.load(v_file)
    
    # Validate shapes
    for key, arr in [('q', q), ('k', k), ('v', v)]:
        if arr.shape != shapes[key]:
            raise ValueError(f"{key} shape mismatch: expected {shapes[key]}, got {arr.shape}")

    q_torch = torch.from_numpy(q.astype(np.float32)).contiguous()
    k_torch = torch.from_numpy(k.astype(np.float32)).contiguous()
    v_torch = torch.from_numpy(v.astype(np.float32)).contiguous()

    return q_torch, k_torch, v_torch

def main():
    # Parse shapes
    shapes = parse_shapes("TestData\\1024x1024\\shapes.txt")
    print("Parsed shapes:", shapes)
    
    # Load matrices
    q, k, v = load_matrices(shapes, "TestData\\1024x1024\\q.npy", "TestData\\1024x1024\\k.npy", "TestData\\1024x1024\\v.npy")
    print(f"Loaded q: {q.shape}, k: {k.shape}, v: {v.shape}")
    
    # measure time taken in torch
    start_time = time.perf_counter()
    out_pytorch = attention_pytorch(
        q,
        k,
        v,
        heads=shapes['q'][1],  # 24
        mask=None,
        attn_precision=None,
        skip_reshape=True,
        skip_output_reshape=True  # Ensure [batch, heads, seq_len, dim_head]
    )
    pytorch_time = (time.perf_counter() - start_time) * 1000
    print(f"PyTorch (ComfyUI) time: {pytorch_time:.2f} ms")
    print(f"PyTorch output shape: {out_pytorch.shape}")
    
    # Compare (optional)
    #np.testing.assert_allclose(out, out_torch.numpy(), rtol=1e-3, atol=1e-3)
    #print("Outputs match within tolerance")

if __name__ == "__main__":
    main()