from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """ learning rate scheduler used in the transformer
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Scaling factor is implemented as in
        http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
    """

    def __init__(
            self, optimizer, d_model, warmup_steps, tot_step, scale,
            last_epoch=-1
            ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.tot_step = tot_step
        self.scale = scale
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch = max(1, self.last_epoch)
        step_num = self.last_epoch
        val = self.scale * self.d_model ** (-0.5) * \
            min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))

        return [base_lr / base_lr * val for base_lr in self.base_lrs]
