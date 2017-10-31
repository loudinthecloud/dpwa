---
- nodes:
<<<nodes>>>

# The probability of initiating a fetch parameters request
- fetch_probability: 1

# The timeout value is used for flow-control
- timeout_ms: 2500

# Choose interpolation method: clock, loss or constant
- interpolation: constant

# Individual interpolation methods configuration:

- constant: { value: 0.5 }

- clock: 0

- loss: 0
