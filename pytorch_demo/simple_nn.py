import torch

X = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]])

y_true = torch.tensor([[4.0],
                       [5.0],
                       [6.0],
                       [7.0]])

class SimpleNeuralNetwork:
    def __init__(self, initial_weight, initial_bias):
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias

    def train(self, learning_rate, epochs):
        w = torch.randn(self.initial_weight, requires_grad=True)
        b = torch.randn(self.initial_bias, requires_grad=True)

        for epoch in range(epochs):
            y_hat = X @ w + b
            loss = torch.mean((y_hat - y_true) ** 2)
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad.zero_()
            b.grad.zero_()

            if epoch % 10 == 0:
                print(f"Epoch {epoch:02d}: Loss {loss.item():.4f}, W = {w.item():.3f}, B = {b.item():.3f}")
            
        print("Weight =", w.item())
        print("Bias =", b.item())

if __name__ == '__main__':
    nn = SimpleNeuralNetwork((1,1), (1,1))
    nn.train(0.01, 100)

