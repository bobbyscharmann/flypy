import torch

# Create a few tensors on GPU
V1 = torch.tensor([1.0, 2.0], requires_grad=True, device='cuda:0')
V2 = torch.tensor([1.0, 1.0], requires_grad=False, device='cuda:0')
V3 = torch.tensor([1.0, 1.0], requires_grad=True, device='cuda:0')

v_sum = 2 * (V1 + V2 + V3)
sum = (v_sum * 3).sum()
print(f"Sum is: {sum}")

# Compute Gradient
a = sum.backward()
print(f"V1.grad Gradient: {V1.grad}")
print(f"V2.grad Gradient: {V2.grad}")
print(f"V3.grad Gradient: {V3.grad}")


class BobsNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):

        super().__init__()
        self.pipe = torch.nn.Sequential(torch.nn.Linear(n_inputs, 5),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(5, n_outputs),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.2),
                                        torch.nn.Softmax(dim=1))

    def forward(self, X):
        Y_hat = self.pipe(X)
        return Y_hat



X = torch.FloatTensor([[3, 3], [4, 4]])
model = BobsNN(n_inputs=2, n_outputs=1)
y_pred = model(X)
print(f"Model predictions: {y_pred}")