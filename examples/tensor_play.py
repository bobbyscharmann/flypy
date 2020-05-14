import torch

V1 = torch.tensor([1.0, 2.0], requires_grad=True)
V2 = torch.tensor([1.0, 1.0], requires_grad=False)

v_sum = 2 * (V1 + V2)
sum = (v_sum * 3).sum()
print(f"Sum is: {sum}")

# Compute Gradient
a = sum.backward()
print(f"V1.grad Gradient: {V1.grad}")
print(f"V2.grad Gradient: {V2.grad}")
