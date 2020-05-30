"""This file will train a neural network to learn a 3D function."""
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
                                        torch.nn.Linear(100, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, n_outputs))

    def forward(self, x):
        Y_hat = self.pipe(x)
        return Y_hat

NUM_INPUT_FEATURES = 2
NUM_SAMPLES = 200
num_epochs = 10000
#X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
col1 = np.linspace (-1, 1, NUM_SAMPLES)
col2 = np.linspace (-1, 1, NUM_SAMPLES)
x_vals = np.zeros((NUM_SAMPLES ** 2, NUM_INPUT_FEATURES))
count = 0
for idx1, x1 in enumerate(col1):
    for idx2, x2 in enumerate(col2):
        x_vals[count, 0] = x1
        x_vals[count, 1] = x2
        count += 1

#x_vals = np.vstack((np.linspace(0, 20, 2000), np.linspace(0, 20, 2000))).T
y_vals = np.zeros((NUM_SAMPLES**2, 1))
for idx, val in enumerate(x_vals):
    y_vals[idx] = np.sin(val[0]) * np.cos(val[1])
    y_vals[idx] = np.sin(10*(val[0]**2 + val[1]**2))/10

xscaler = MinMaxScaler()
yscaler = MinMaxScaler()
xscaler.fit(x_vals.reshape(-1, NUM_INPUT_FEATURES))
x_vals_scaled = xscaler.transform(x_vals.reshape(-1, NUM_INPUT_FEATURES))
yscaler.fit(y_vals.reshape(-1, 1))
y_vals_scaled = yscaler.transform(y_vals.reshape(-1, 1))
X_orig = torch.FloatTensor((x_vals_scaled.reshape(-1, NUM_INPUT_FEATURES)))
x = X_orig[:, 0:2]
y = torch.FloatTensor(y_vals_scaled)#X_orig[2, :].T

model = BobsNN(n_inputs=NUM_INPUT_FEATURES, n_outputs=1)#.to(device='cuda:0')
batch_size = NUM_SAMPLES**2
X_train = x
Y_train = y
print(f"X.train{X_train.shape}")
print(f"Y.train{Y_train.shape}")


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
        Y_hat = model(X_batch)
        loss = criterion(Y_hat, Y_batch)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_num}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        # Every ten epochs, test out the model
        if epoch % (num_epochs / 100) == 0:
            y_vals = model(X_train)
            fig = plt.figure(figsize=(13,11))
            ax = fig.add_subplot(111, projection='3d')
            tmp_x1 = x_vals[:, 0]
            tmp_x2 = x_vals[:, 1]
            tmp_y = y_vals.detach().numpy()

            ax.scatter(x_vals[:, 0],
                       x_vals[:, 1],
                       tmp_y,
                       s=10*np.ones(x_vals[:, 0].shape), c=tmp_y,#range(NUM_SAMPLES**2),
                       label="f_hat(X) (model estimate)")
            x1 = X_orig[0, :].reshape(-1, 1)
            x2 = X_orig[1, :].reshape(-1, 1)
            tmp = xscaler.inverse_transform(X_orig)
            ax.scatter(tmp[:, 0], tmp[:, 1], y, label="f(X) (truth data)", alpha=0.1)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"y = sin(10*(x^2 + y^2))/10 @ epoch {epoch}")
            plt.legend()
            plt.savefig(os.path.join("results", f"epoch_{epoch}"))
            plt.close()

# Append all the images into a list for imageio
for file in os.listdir("results"):
    images.append(imageio.imread(os.path.join("results", file)))

# Actually create the GIF
imageio.mimsave("ripple.gif", images, fps=10)

# Clean up all the temporary images - surely there is a better way to do this with imageio without writing out
# all the images, but I struggled to make it work and this was simple
[os.remove(os.path.join("results", file)) for file in os.listdir("results")]

