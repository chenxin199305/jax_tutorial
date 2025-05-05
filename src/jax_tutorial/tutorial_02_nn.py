import jax
import jax.numpy as jnp

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

print(
    f"#" * 20 + "\n",
    f"X.shape: {X.shape} \n",
    f"y.shape: {y.shape} \n",
)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(X)

print(
    f"#" * 20 + "\n",
    f"X.shape: {X.shape} \n",
    f"y.shape: {y.shape} \n",
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(
    f"#" * 20 + "\n",
    f"X_train.shape: {X_train.shape} \n",
    f"y_train.shape: {y_train.shape} \n",
    f"X_test.shape: {X_test.shape} \n",
    f"y_test.shape: {y_test.shape} \n",
)


# input -> hidden layer 1 -> hidden layer 2 -> output
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.params = {
            "W1": jax.random.normal(jax.random.PRNGKey(0), (input_dim, hidden_dim1)),
            "b1": jnp.zeros((hidden_dim1,)),
            "W2": jax.random.normal(jax.random.PRNGKey(1), (hidden_dim1, hidden_dim2)),
            "b2": jnp.zeros((hidden_dim2,)),
            "W3": jax.random.normal(jax.random.PRNGKey(2), (hidden_dim2, output_dim)),
            "b3": jnp.zeros((output_dim,))
        }

    def forward(self, x):
        # Layer 1
        z1 = jnp.dot(x, self.params["W1"]) + self.params["b1"]
        a1 = jax.nn.relu(z1)

        # Layer 2
        z2 = jnp.dot(a1, self.params["W2"]) + self.params["b2"]
        a2 = jax.nn.relu(z2)

        # Output layer
        z3 = jnp.dot(a2, self.params["W3"]) + self.params["b3"]
        return z3

    def predict(self, x):
        logits = self.forward(x)
        return jnp.argmax(logits, axis=1)


# Loss function
def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=1))


# Gradient function
def compute_gradients(model, x, y):
    def loss_fn(params):
        logits = model.forward(x)
        return cross_entropy_loss(logits, y)

    grads = jax.grad(loss_fn)(model.params)
    return grads


# Update parameters
def update_params(params, grads, learning_rate=0.01):
    for key in params.keys():
        params[key] -= learning_rate * grads[key]
    return params


# Training function
def train(model, x_train, y_train, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        grads = compute_gradients(model, x_train, y_train)
        model.params = update_params(model.params, grads, learning_rate)
        if epoch % 100 == 0:
            logits = model.forward(x_train)
            loss = cross_entropy_loss(logits, y_train)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Initialize the model
input_dim = X_train.shape[1]
hidden_dim1 = 10
hidden_dim2 = 10
output_dim = y_train.shape[1]
model = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Train the model
train(model, X_train, y_train, epochs=1000, learning_rate=0.001)

# Evaluate the model
logits = model.forward(X_test)
