# Optimizer

When using an optimizer, you need to configure the `name` item to indicate which optimizer to use, and then configure their respective parameters according to different optimizer. The following are the names of each optimizer.

| Optimizer | name    |
| --------- | ------- |
| SGD       | sgd     |
| FTRL      | ftrl    |
| Adagrad   | adagrad |
| Adam      | adam    |
| AdamW     | adamw   |
| Lion      | lion    |

## SGD

[SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) configure the following parameters:

1. $\gamma$: learning rate, default: 1e-3, **configure key**: `gamma`
2. $\lambda$: weight decay, default: 0, **configure key**: `lambda`

## FTRL

[FTRL](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37013.pdf) configure the following parameters:

1. $\alpha$: learning rate, default: 5e-3, **configure key**: `gamma`
2. $\beta:$\\beta$ param, default: 0.0, **configure key**: `beta`
3. $\lambda_1$: L1 regulation, default: 0.0, **configure key**: `lambda1`
4. $\lambda_2$: L2 regulation, default: 0.0, **configure key**: `lambda2`

## Adagrad

[Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad) configure the following parameters:

1. $\gamma$: learning rate, default: 1e-2, **configure key**: `gamma`
2. $\lambda$: weight decay, default: 0.0, **configure key**: `lambda`
3. $\eta$: learning rate decay, default: 0.0, **configure key**: `eta`
4. $\epsilon$: minimun error term, default: 1e-10, **configure key**: `epsilon`

## Adam

[Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) configure the following parameters(not support amsgrad):

1. $\gamma$: learning rate, default: 1e-3, **configure key**: `gamma`
2. $\beta_1$: moving averages of gradient coefficient, default: 0.9, **configure key**: `beta1`
3. $\beta_2$: moving averages of gradient's square coefficient, default: 0.999, **configure key**: `beta2`
4. $\lambda$: weight decay rate, default: 0.0, **configure key**: `lambda`
5. $\epsilon$: minimun error term, default: 1e-8, **configure key**: `epsilon`

## AdamW

[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) configure the following parameters(not support amsgrad):

1. $\gamma$: learning rate, default: 1e-3, **configure key**: `gamma`
2. $\beta_1$: moving averages of gradient coefficient, default: 0.9, **configure key**: `beta1`
3. $\beta_2$: moving averages of gradient's square coefficient, default: 0.999, **configure key**: `beta2`
4. $\lambda$: weight decay rate, default: 1e-2, **configure key**: `lambda`
5. $\epsilon$: minimun error term, default: 1e-8, **configure key**: `epsilon`

## Lion

[Lion](https://arxiv.org/abs/2302.06675) configure the following parameters:

1. $\eta$: learing rate, default: 3e-4, **configure key**: `eta`
2. $\beta_1$: moving averages of gradient coefficient, default: 0.9, **configure key**: `beta1`
3. $\beta_2$: moving averages of gradient's square coefficient, default: 0.99, **configure key**: `beta2`
4. $\lambda$: weight decay, default: 0.01, **configure key**: `lambda`

## Example

```python
adagrad_optimizer = {
    "name": "adagrad",
    "eta": 0.0,
    "gamma": 0.001,
    "lambda": 0.0,
    "epsilon": 1e-8,
}

adam_optimizer = {
    "name": "adam",
    "gamma": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "lambda": 0.0,
    "epsilon": 1e-8,
}
```