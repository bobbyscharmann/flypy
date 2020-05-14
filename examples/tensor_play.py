import torch

V1 = torch.tensor([2.0, 2.0], requires_grad=True)
V2 = torch.tensor([1.0, 1.0], requires_grad=True)

v_sum = V1 + V2
sum = (v_sum * 2).sum()
print(f"Sum is: {sum}")

# Compute Gradient
a = sum.backward()
print(f"V1.grad Gradient: {V1.grad}")
print(f"V2.grad Gradient: {V2.grad}")
