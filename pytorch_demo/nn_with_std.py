import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(4,2)

y = torch.tensor([[4.0],
                  [5.0],
                  [6.0],
                  [7.0]])


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_feature, hidden_feature):
        super().__init__()

        self.layer_one = nn.Linear(in_feature, hidden_feature)
        self.activation = nn.ReLU()
        self.layer_two = nn.Linear(hidden_feature, 1)

    def forward(self, x):
        x = self.layer_one(x)
        x = self.activation(x)
        x = self.layer_two(x)
        return x


class Training:
    
    def train(self, lr):
        model = FeedForwardNetwork(2,8)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = lr)
        epochs = 1000

        for epoch in range(epochs):
            y_hat = model(X)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch:02d}  Loss = {loss.item():.4f}")

        print("\nLayer 1 Weights:")
        print(model.layer_one.weight)

        print("\nLayer 1 Bias:")
        print(model.layer_one.bias)

        print("\nLayer 2 Weights:")
        print(model.layer_two.weight)

        print("\nLayer 2 Bias:")
        print(model.layer_two.bias)

        # print(model.state_dict())
        print(X)

        with torch.no_grad(): 
            predictions = model(X)
            print("\n--- Predictions vs Targets ---")
            for i in range(len(y)):
                print(f"Input: {X[i].numpy()} -> Predicted: {predictions[i].item():.4f} | Target: {y[i].item():.4f}")

            
if __name__ == '__main__':
    train = Training()
    train.train(0.01)
    

            


