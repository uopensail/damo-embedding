# Optimizer

When using an optimizer, you need to configure the `name` item to indicate which optimizer to use, and then configure their respective parameters according to different optimizer. The following are the names of each optimizer.

| Optimizer | name    |
| --------- | ------- |
| AdamW     | adamw   |


## AdamW

AdamW is an optimization algorithm that is widely used in training neural networks. It is an extension of the Adam optimizer, with a key modification in how weight decay is handled.

### Overview

- **Adam Optimizer**: Adam combines the advantages of two other extensions of stochastic gradient descent — Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). Adam maintains per-parameter learning rates that are adapted based on the first and second moments of the gradients.

- **Weight Decay**: In the original Adam optimizer, weight decay (often used as L2 regularization) is implemented by adding a penalty to the loss function. However, this approach can interfere with the adaptive learning rate mechanism of Adam.

- **AdamW Modification**: AdamW decouples weight decay from the gradient-based optimization step. Instead of applying weight decay through the loss function, it directly subtracts the weight decay term proportional to the learning rate and weights during the parameter update. This leads to more effective regularization and often results in better generalization performance.

- **Benefits**: By decoupling weight decay from the optimization step, AdamW achieves a better balance between convergence speed and model generalization. This makes it particularly effective for training large, complex models.

- **Widespread Use**: AdamW has become a popular choice in the machine learning community due to its robust performance across a variety of tasks and architectures. It is now considered a go-to optimizer for many deep learning practitioners.

AdamW's ability to provide a more principled approach to weight decay makes it a preferred choice in both academic research and industry applications, where model performance and generalization are critical.

### Configuration
[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) configure the following parameters(not support amsgrad):

1. $\gamma$: learning rate, default: 1e-3, **configure key**: `gamma`
2. $\beta_1$: moving averages of gradient coefficient, default: 0.9, **configure key**: `beta1`
3. $\beta_2$: moving averages of gradient's square coefficient, default: 0.999, **configure key**: `beta2`
4. $\lambda$: weight decay rate, default: 1e-2, **configure key**: `lambda`
5. $\epsilon$: minimun error term, default: 1e-8, **configure key**: `epsilon`


## Example

```python

adamw_optimizer = {
    "name": "adamw",
    "gamma": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "lambda": 0.0,
    "epsilon": 1e-8,
}
```