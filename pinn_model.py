import torch
import torch.nn as nn

# Simple neural network
class SoilMoisturePINN(nn.Module):
    def _init_(self):
        super()._init_()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


# Data loss (difference between prediction & actual)
def data_loss_fn(predicted, actual):
    return torch.mean((predicted - actual) ** 2)


# Physics loss (rule-based)
def physics_loss_fn(temp, rain, predicted):
    # Rule: If temperature is high & no rain,
    # soil moisture should not increase
    violation = torch.relu(predicted * (temp > 35) * (rain == 0))
    return torch.mean(violation)


# Total loss = Data loss + Physics loss
def total_loss(predicted, actual, temp, rain):
    data_loss = data_loss_fn(predicted, actual)
    physics_loss = physics_loss_fn(temp, rain, predicted)
    return data_loss + physics_loss