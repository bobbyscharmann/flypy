import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import imageio
import os
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

np.random.seed(42)
class BobsNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):

        super(BobsNN, self).__init__()
        self.pipe = torch.nn.Sequential(torch.nn.Linear(n_inputs, 100),
                                        #torch.nn.clamp(min=0),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(100, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(32, n_outputs),
                                        #torch.nn.ReLU(),
                                        )

    def forward(self, x):
        Y_hat = self.pipe(x)
        return Y_hat

x_vals = np.linspace(0, 1, 2000)
X = torch.FloatTensor([x_vals, x_vals**3 + 2*x_vals**2+3 + 1 * np.random.uniform(0, 0.1, 2000)])
#X = X / X.max(0, keepdim=True)[0]
x = X[0, :].T #torch.randn(2000, 1)
y = X[1, :].T#torch.randn(2000, 1)
#x = x / x.max(0, keepdim=True)[0]
#x = torch.randn(2000, 1)
#y = torch.randn(2000, 1)
#X = x
print(f"X.shape{X.shape}")
model = BobsNN(n_inputs=1, n_outputs=1)#.to(device='cuda:0')
#model = TwoLayerNet(1, 100, 1)
#print(f"Model predictions: {y_pred}")
batch_size = 2000
num_epochs = 1000
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
print(f"X.train{X_train.shape}")
print(f"Y.train{Y_train.shape}")
print(f"X.test{X_test.shape}")
print(f"Y.test{Y_test.shape}")


optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss(reduction='mean')
images = []
if not os.path.exists("results"):
    os.makedirs("results", 777)
for epoch in range(num_epochs):
    i = 0
    for batch in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        X_batch = X_train[batch:batch+batch_size]
        Y_batch = Y_train[batch:batch+batch_size]
        Y_train = Y_train.view(Y_train.shape[0], -1)
        Y_hat = model(X_batch.view(X_batch.shape[0], -1))
        loss = criterion(Y_hat, Y_batch)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            x_vals = torch.FloatTensor(np.linspace(0, 1, 2000))
            y_vals = model(x_vals.view(x_vals.shape[0], -1))
            fig = plt.figure()
            plt.scatter(x_vals.detach().numpy(), y_vals.detach().numpy(), label="f_hat(X) (model estimate)")
            plt.scatter(x, y, label="f(X) (truth data)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"y=x^3 + 2x^2 + 3 @ epoch {epoch}")
            plt.legend()
            plt.savefig(os.path.join("results", f"epoch_{epoch}"))
            plt.close()

for file in os.listdir("results"):
    images.append(imageio.imread(os.path.join("results", file)))
imageio.mimsave("replay.gif", images, fps=10)
[os.remove(os.path.join("results", file)) for file in os.listdir("results")]

y = model(torch.FloatTensor([1]))
print(f"model prediction: {y}")
y = model(torch.FloatTensor([3]))
print(f"model prediction: {y}")
