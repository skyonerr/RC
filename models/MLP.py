"""Reservoir readout with a conductance-parameterized first layer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aggregate_states(states: np.ndarray, mode: str = "last") -> np.ndarray:
    """Select the final state or compute the temporal mean."""
    if states.ndim != 3:
        raise ValueError(f"states 维度应为 (N, W, D)，当前为 {states.shape}")
    if mode == "last":
        return states[:, -1, :]
    elif mode == "mean":
        return states.mean(axis=1)
    else:
        raise ValueError("mode 必须是 'last' 或 'mean'")


class PhysicalConductanceLayer(nn.Module):
    """Linear layer with sigmoid-bounded positive weights and an unconstrained bias."""

    def __init__(self, in_features, out_features,
                 init_R0=50.0):
        """Set conductance bounds from the reference resistance in ohms."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        G0_ref = 1.0 / init_R0

        alpha = 2.0
        self.G0_min = G0_ref / alpha
        self.G0_max = G0_ref * alpha

        self.G0_param = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, V_in):
        u = torch.sigmoid(self.G0_param)
        real_G0 = self.G0_min + (self.G0_max - self.G0_min) * u


        I_out = F.linear(V_in, real_G0) + self.bias
        return I_out


class PhysicsNet(nn.Module):
    """Conductance layer, LeakyReLU, linear readout, and output ReLU."""

    def __init__(self, input_dim, hidden_dim=64, base_R=50.0):
        super(PhysicsNet, self).__init__()

        self.layer1 = PhysicalConductanceLayer(
            in_features=input_dim,
            out_features=hidden_dim,
            init_R0=base_R
        )

        self.act = nn.LeakyReLU(negative_slope=0.1)

        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        currents = self.layer1(x)
        a1 = self.act(currents)
        out = self.layer2(a1)

        return F.relu(out)


class MLPReadout:
    """Full-batch Adam training and inference for reservoir state features."""

    def __init__(
            self,
            hidden_layer_sizes=(64,),
            alpha=1e-3,
            max_iter=1000,
            aggregation="last",
            random_state=42,
            verbose=False,
            device_base_resistance: float = 50.0
    ):
        self.hidden_dim = hidden_layer_sizes[0] if isinstance(hidden_layer_sizes, tuple) else hidden_layer_sizes
        self.lr = 0.001
        self.weight_decay = alpha
        self.epochs = max_iter
        self.aggregation = aggregation
        self.verbose = verbose
        self.random_state = random_state

        self.base_R = device_base_resistance
        self.model = None
        self.initial_G0 = None
        self.loss_history = []

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(self, states: np.ndarray, y: np.ndarray):
        """Fit damage targets from reservoir states shaped (samples, steps, features)."""
        X_feat = aggregate_states(states, mode=self.aggregation)
        y = np.asarray(y).reshape(-1, 1)

        X_tensor = torch.tensor(X_feat, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)

        input_dim = X_feat.shape[1]

        self.model = PhysicsNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            base_R=self.base_R
        ).to(DEVICE)

        # Record the initial conductances for inspection.
        with torch.no_grad():
            u = torch.sigmoid(self.model.layer1.G0_param)
            min_g = self.model.layer1.G0_min
            max_g = self.model.layer1.G0_max
            self.initial_G0 = (min_g + (max_g - min_g) * u).cpu().numpy().copy()

        if self.verbose:
            print(f"[Init G0] Saved initial weight matrix shape: {self.initial_G0.shape}")

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        self.model.train()
        self.loss_history = []
        for epoch in range(self.epochs):
            y_pred = self.model(X_tensor)
            loss = criterion(y_pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

            if self.verbose and (epoch % 100 == 0):
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict nonnegative damage from reservoir states."""
        if self.model is None:
            raise ValueError("Model not fitted.")

        X_feat = aggregate_states(states, mode=self.aggregation)
        X_tensor = torch.tensor(X_feat, dtype=torch.float32).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()
