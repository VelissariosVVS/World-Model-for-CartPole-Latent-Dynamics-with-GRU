```markdown 
``` 
# 🧠 World Model for CartPole

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A latent dynamics world model for the CartPole environment, implementing an encoder-decoder architecture with GRU-based state prediction. This project demonstrates core concepts from [Ha & Schmidhuber's World Models](https://worldmodels.github.io/) paper.

## 🎯 Project Goal

Learn a **latent dynamics model** that can:
1. **Encode** observations into a compact latent space
2. **Predict** future latent states given actions  
3. **Decode** latent states back to observations

The model is trained entirely on **sequences of observations and actions** collected from random, heuristic, and PD control policies.

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Observation │────▶│   Encoder   │────▶│   Latent    │
│    x_t      │     │             │     │     z_t     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                 │
                                            (GRU cell)
                                                 │
┌─────────────┐     ┌─────────────┐     ┌──────▼──────┐
│ Prediction  │◀────│   Decoder   │◀────│   z_{t+1}   │
│    x̂_{t+1}  │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Components:
- **Encoder**: 3-layer MLP (4 → 64 → 32 → latent_dim)
- **Decoder**: 3-layer MLP (latent_dim → 32 → 64 → 4)
- **LatentDynamics**: GRUCell that predicts next latent state from (z_t, a_t)

## 📊 Dataset

- **10,000 trajectories** collected from CartPole-v1
- **Mixed policy**: 10% PD control, 45% heuristic, 45% random
- **Sequence length**: 20 steps per training example
- **Train/val split**: 80/20 by trajectory

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/world-model-cartpole.git
cd world-model-cartpole
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate data
Run the Jupyter notebook:
```bash
jupyter notebook notebooks/data_preprocessing.ipynb
```

### 4. Train the world model
```bash
python main.py
# Select option 1
```

### 5. Evaluate the model
```bash
python main.py
# Select option 2
```

## 📈 Results

| Horizon | RMSE (full state) |
|---------|-------------------|
| 1 step  | 0.08 |
| 5 steps | 0.21 |
| 10 steps| 0.38 |
| 30 steps| 0.67 |

### Latent Space Visualization
Latent states colored by pole angle show meaningful clustering.

### Multi-step Rollout
The model maintains reasonable predictions for ~10-15 steps before divergence.

## 📁 Project Structure

```
world-model-cartpole/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── data_preprocessing.ipynb
├── src/
│   ├── __init__.py
│   ├── environment.py
│   ├── trajectory.py
│   ├── trajectory_generation.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── latent_dynamics.py
│   ├── train_autoencoder.py
│   ├── evaluation_world_model.py
│   └── main.py
├── plots/               (generated during evaluation)
└── processed/           (saved .pt files)
```

## 🔮 Future Work

This is a **foundational world model**. Next steps to complete the system:

1. **Planning in Latent Space**  
   Implement Cross-Entropy Method (CEM) or Model Predictive Control (MPC) to select optimal actions by simulating rollouts in the latent space.

2. **Policy Training (Dreamer-style)**  
   Train a policy network using imagined rollouts from the world model, enabling latent imagination-based learning.

3. **Vision-based Extension**  
   Replace low-dimensional state input with pixel observations using a CNN encoder (e.g., VAE).

4. **Uncertainty Estimation**  
   Add ensembles or probabilistic dynamics to capture epistemic uncertainty.

## 📚 References

- [Ha & Schmidhuber (2018) - World Models](https://worldmodels.github.io/)
- [Hafner et al. (2019) - Dreamer](https://arxiv.org/abs/1912.01603)
- [OpenAI Gym - CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## 📝 License

MIT

## 👤 Author

Stravomitis Velissarios Visarion - VelissariosVVS
