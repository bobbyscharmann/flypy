import numpy as np
import torch
from sklearn.model_selection import train_test_split
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

        super(BobsNN, self).__init__()
        self.pipe = torch.nn.Sequential(torch.nn.Linear(n_inputs, 5),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(5, n_outputs),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.2),
                                        torch.nn.Softmax(dim=1),
                                        )

    def forward(self, X):
        Y_hat = self.pipe(X)
        return Y_hat

x_vals = np.linspace(0,200,1)
X = torch.tensor([[x_vals], [x_vals**2]])
model = BobsNN(n_inputs=1, n_outputs=1)#.to(device='cuda:0')
#print(f"Model predictions: {y_pred}")
batch_size = 2
num_epochs = 10
X_train, X_test, Y_train, Y_test = train_test_split(X[:, 0], X[:, 1], test_size=0.2)


optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
for epoch in range(num_epochs):
    for batch in range(1, len(X_train), batch_size):
        X_batch = X_train[batch:batch+batch_size]
        Y_batch = Y_train[batch:batch+batch_size]
        Y_hat = model(X_batch)
        loss = criterion(Y_batch, Y_hat)
        print(f"Loss: {loss}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



