---
- nodes:
<<<nodes>>>

# The probability of initiating a parameter pull request
- pull_probability: 1

# The timeout value is used for flow-control
- timeout_ms: 2500

# Choose interpolation method: clock, loss, loss_divergence, constant or linear
- interpolation: loss_divergence

# Individual interpolation methods configuration:

- constant: { value: 0.5 }

- linear: { start: 0.5, end: 0.05, target: 5000 }

- clock: 0

- loss: 0

- loss_divergence: { target_divergence_loss: 0.2 }
