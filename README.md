# The Hierarchical Transformer-Boost RL Framework for Dynamic Multi-Asset Portfolio Optimization

## 📦 Installation

To install the required dependencies, follow these steps:

```bash
# Clone the repository
git clone https://github.com/ramiredddy/The-Heirarchial-Transformer-Boost-RL-Framework-for-Dynamic-Multi-Asset-Portfolio-Optimization.git
cd The-Heirarchial-Transformer-Boost-RL-Framework-for-Dynamic-Multi-Asset-Portfolio-Optimization

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

```yaml
learning_rate: 3e-4
batch_size: 256
gamma: 0.99
epsilon_clip: 0.2
entropy_coef: 0.01
value_loss_coef: 0.5
max_grad_norm: 0.5
transformer_layers: 6
attention_heads: 8
hidden_dim: 512
```

## 📊 Methodology

This framework combines three powerful approaches:

1. **Hierarchical Transformer Architecture**: Multi-head self-attention mechanisms capture complex temporal dependencies in financial time series data
2. **Gradient Boosting Integration**: XGBoost enhances feature extraction and provides robust prediction signals
3. **Proximal Policy Optimization (PPO)**: Stable reinforcement learning algorithm for continuous action spaces in portfolio allocation

The model processes historical price data, technical indicators, and market microstructure features through transformer layers, then uses PPO to optimize portfolio weights while maintaining transaction cost awareness.

## 🏛️ Architecture Overview

```
[Market Data] --> [Feature Engineering] --> [Transformer Encoder]
                                                     |
                                                     v
                                          [Attention Layers (x6)]
                                                     |
                                                     v
                                          [XGBoost Feature Boost]
                                                     |
                                                     v
                                             [PPO Actor-Critic]
                                                     |
                                    +----------------+----------------+
                                    |                                 |
                                [Actor]                           [Critic]
                            (Policy Network)                (Value Network)
                                    |
                                    v
                            [Portfolio Weights]
```

The architecture consists of multiple attention heads that process temporal sequences, followed by gradient boosting for feature enhancement, and finally a PPO-based actor-critic network for action generation.

## 🎯 Evaluation Metrics

The framework performance is evaluated using the following key metrics:

- **Sharpe Ratio**: Risk-adjusted returns measuring excess return per unit of volatility
- **Maximum Drawdown**: Largest peak-to-trough decline during the investment period
- **Cumulative Returns**: Total portfolio return over the evaluation period
- **Win Rate**: Percentage of profitable trading periods
- **Sortino Ratio**: Downside risk-adjusted return (penalizes only negative volatility)
- **Information Ratio**: Active return relative to benchmark divided by tracking error
- **Calmar Ratio**: Annual return divided by maximum drawdown

All metrics are computed on out-of-sample test data to ensure robust evaluation.

## 🚀 Usage Example

Here's a quick example to get you started:

```python
import numpy as np
from portfolio_optimizer import HierarchicalTransformerPPO

# Initialize the model
model = HierarchicalTransformerPPO(
    n_assets=10,
    learning_rate=3e-4,
    transformer_layers=6,
    attention_heads=8
)

# Load your market data
market_data = np.load('market_data.npy')  # Shape: (timesteps, n_assets, features)

# Train the model
model.train(
    data=market_data,
    epochs=100,
    batch_size=256
)

# Generate portfolio weights for new data
new_data = np.load('test_data.npy')
portfolio_weights = model.predict(new_data)

print(f"Optimal portfolio weights: {portfolio_weights}")
```

## 🧪 Future Work
- [ ] Integration with real-time trading APIs
- [ ] Multi-objective optimization with Pareto frontiers
- [ ] Transfer learning across different market regimes
- [ ] Explainable AI for decision interpretation
- [ ] Distributed training for larger asset universes

## 📚 References
1. Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS.
2. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv.
3. Jiang, Z., et al. (2017). *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem*. arXiv.
4. Zhang, Z., et al. (2020). *Deep Reinforcement Learning for Trading*. Journal of Financial Data Science.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commit messages
3. **Add tests** if you're adding new functionality
4. **Ensure the test suite passes** before submitting
5. **Submit a Pull Request** with a comprehensive description of changes

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add docstrings to all functions and classes
- Update documentation for any changed functionality
- For major changes, please open an issue first to discuss your proposal
- Be respectful and constructive in all interactions

### Areas for Contribution

- Bug fixes and issue resolutions
- Performance optimizations
- Documentation improvements
- New feature implementations
- Test coverage expansion
- Example notebooks and tutorials

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author
**Rami Reddy**
- GitHub: [@ramiredddy](https://github.com/ramiredddy)

## 🙏 Acknowledgments
- Thanks to the open-source community for PyTorch and Gym/Gymnasium
- Inspired by recent advances in transformers and deep reinforcement learning
- Financial data providers and research institutions

---
⭐ If you find this project useful, please consider giving it a star!
