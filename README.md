[![Pytest and Linter](https://github.com/asteppke/tinypid/actions/workflows/python-app.yml/badge.svg)](https://github.com/asteppke/tinypid/actions/workflows/python-app.yml)

# Tiny PID
Minimal PID controller in Python. 

Optional features: 
- Output limiting
- Anti-windup mechanism
- Lowpass filtering of derivative component
- Bumpless transfer between manual and automatic control
- Gain scheduling
- Feed forward component
- Output rate limiting
- Output dead-band (hold the previous output for small changes, e.g. to reduce motor noise)
- Output dithering (add small pseudo-random noise to improve effective resolution on quantized actuators)

![PID control example](examples/control_example.png)

## Future ideas
- Beyond gain scheduling also schedule, limits, and anti-windup
