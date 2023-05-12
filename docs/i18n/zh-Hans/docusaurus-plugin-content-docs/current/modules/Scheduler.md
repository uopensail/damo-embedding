# Learning Rate Scheduler

When training deep neural networks, it is often useful to reduce learning rate as the training progresses. Learning rate schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule. 

When using an scheduler, you need to configure the `name` item to indicate which scheduler to use, and then configure their respective parameters according to different scheduler. The following are the names of each scheduler. If name is empty, no scheduler is used.

| scheduler                | name                     |
|:------------------------:|:------------------------:|
| Exponential Decay        | exponential_decay        |
| Polynomial Decay         | polynomial_decay         |
| Nature Exponential Decay | nature_exponential_decay |
| Inverse Time Decay       | inverse_time_decay       |
| Cosine Decay             | cosine_decay             |
| Liner Cosine Decay       | liner_cosine_decay       |

## Exponential Decay

$learning\_rate * decay\_rate ^{\frac{global\_step}{decay\_steps}}$

configure the following parameters:

1. **decay_steps**: float type

2. **decay_rate:** float type

## Polynomial Decay

$(learning\_rate - end\_learning\_rate)*decay\_rate^{1.0 - \frac{min(global\_step, decay\_steps)}{decay\_steps}} + end\_learning\_rate$

configure the following parameters:

1. **decay_steps**: float type

2. **decay_rate**: float type, default: 1e-3

3. **end_learning_rate**: float type, default: 1.0

## Nature Exponential Decay

$learning\_rate*e^{-decay\_rate *{\frac{global\_step}{decay\_steps}}}$

configure the following parameters:

1. **decay_steps**: float type

2. **decay_rate:** float type

## Inverse Time Decay

$\frac{learning\_rate}{1.0+ decay\_rate *{\frac{global\_step}{decay\_steps}}}$

configure the following parameters:

1. **decay_steps**: float type

2. **decay_rate:** float type

## Cosine Decay

$learning\_rate * 0.5 *(1.0 + cos(\pi*\frac{global\_step}{decay\_steps})$

configure the following parameters:

1. **decay_steps**: float type

## Liner Cosine Decay

$liner\_decay = \frac{decay\_steps - min(global\_step, decay\_steps)}{decay\_steps}$

$cos\_decay = -0.5 * (1.0 + cos(2\pi*num\_periods*\frac{min(global\_step, decay\_steps)}{decay\_steps})$

$learning\_rate * (\alpha + liner\_decay)*cos\_decay+\beta$

configure the following parameters:

1. **alpha**: $\alpha$, float type, default: 0.0

2. **beta**: $\beta$, float type, default: 1e-3

3. **num_periods**: float type, default: 0.5

4. **decay_steps**: float type