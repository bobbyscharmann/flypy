import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set the seed for determinism
np.random.seed(42)

# Custom NN architecture
class BobsNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):

        super(BobsNN, self).__init__()
        self.pipe = torch.nn.Sequential(torch.nn.Linear(n_inputs, 100),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(100, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(32, n_outputs),
                                        )

    def forward(self, x):
        Y_hat = self.pipe(x)
        return Y_hat

x_vals = np.vstack((np.linspace(0, 20, 2000), np.linspace(0, 20, 2000))).T
#y_vals = np.zeros(x_vals.shape)
#for idx, x in enumerate(x_vals):
#    if idx < len(x_vals) / 3:
#        y_vals[idx] = np.sin(x)
#    elif idx > 2 * len(x_vals) / 3:
#        y_vals[idx] = 1
#    else:
#        y_vals[idx] = np.cos(x)

y_vals = np.sin(x_vals)
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()
xscaler.fit(x_vals.reshape(-1, 2))
x_vals_scaled = xscaler.transform(x_vals.reshape(-1, 2))
yscaler.fit(y_vals.reshape(-1, 1))
y_vals_scaled = yscaler.transform(y_vals.reshape(-1, 2))
X_orig = torch.FloatTensor([x_vals_scaled.reshape(2, -1), y_vals_scaled.reshape(2, -1)])
x = X_orig[0:1, :].T
y = torch.FloatTensor(y_vals_scaled)#X_orig[2, :].T

model = BobsNN(n_inputs=2, n_outputs=1)#.to(device='cuda:0')
batch_size = 2000
num_epochs = 100
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
print(f"X.train{X_train.shape}")
print(f"Y.train{Y_train.shape}")
print(f"X.test{X_test.shape}")
print(f"Y.test{Y_test.shape}")


optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss(reduction='mean')
images = []

# Create the results folder if it does not exist. This folder will store the images of model results at each epoch
if not os.path.exists("results"):
    os.makedirs("results", 777)
for epoch in range(num_epochs):
    i = 0
    batch_num = 0
    # TODO: You really should shuffle between batches
    for batch in range(0, len(X_train), batch_size):
        batch_num += 1
        optimizer.zero_grad()
        X_batch = X_train[batch:batch+batch_size]
        Y_batch = Y_train[batch:batch+batch_size]
        Y_train = Y_train.view(Y_train.shape[0], -1)
        Y_hat = model(X_batch.view(X_batch.shape[0], -1))
        loss = criterion(Y_hat, Y_batch)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_num}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        # Every ten epochs, test out the model
        if epoch % 10 == 0:
            x_vals = torch.FloatTensor(np.linspace(0, 1, 2000))
            x_vals = np.vstack((np.linspace(0, 20, 2000), np.linspace(0, 20, 2000))).T
            x_vals = x_vals.view(dtype=float)
            y_vals = model(torch.FloatTensor(x_vals))
            fig = plt.figure()
            tmp_x1 = xscaler.inverse_transform(x_vals.reshape(-1, 2))[:, 0]
            tmp_x2 = xscaler.inverse_transform(x_vals.reshape(-1, 2))[:, 1]
            plt.scatter(tmp_x1,
                        tmp_x2,
                        yscaler.inverse_transform(y_vals.detach().numpy().reshape(-1, 1)),
                        label="f_hat(X) (model estimate)")
            x1 = X_orig[0, :].T
            x2 = X_orig[1, :].T
            y = X_orig[2, :].T
            plt.scatter(xscaler.inverse_transform(x1),
                        xscaler.inverse_transform(x2),
                        yscaler.inverse_transform(y), label="f(X) (truth data)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"y=sin(x) @ epoch {epoch}")
            plt.legend()
            plt.savefig(os.path.join("results", f"epoch_{epoch}"))
            plt.close()

# Append all the images into a list for imageio
for file in os.listdir("results"):
    images.append(imageio.imread(os.path.join("results", file)))

# Actually create the GIF
imageio.mimsave("replay.gif", images, fps=10)

# Clean up all the temporary images - surely there is a better way to do this with imageio without writing out
# all the images, but I struggled to make it work and this was simple
[os.remove(os.path.join("results", file)) for file in os.listdir("results")]

