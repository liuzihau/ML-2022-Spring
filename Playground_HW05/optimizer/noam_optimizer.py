class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * get_rate(self.model_size, step, self.warmup)


def get_rate(d_model, step_num, warmup_step):
    """
    Optimizer: Adam + lr scheduling
    Inverse square root scheduling is important to the stability when training Transformer.
    It's later used on RNN as well. Update the learning rate according to the following equation.
    Linearly increase the first stage, then decay proportionally to the inverse square root of time step.
    lrate=d−0.5model⋅min(step_num−0.5,step_num⋅warmup_steps−1.5)
    """
    # TODO: Change lr from constant to the equation shown above
    lr = d_model ** (-0.5) * min(step_num ** -0.5, step_num * (warmup_step ** -1.5))
    return lr
