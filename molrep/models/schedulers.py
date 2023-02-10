
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_lr_scheduler(optimizer, model_configs, num_samples=0):
    sched_dict = model_configs['scheduler']
    if sched_dict is None:
        return None

    sched_s = sched_dict['class']
    if sched_s == 'ECCLR':
        return ECCLR(optimizer, model_configs)
    elif sched_s == 'NoamLR':
        return NoamLR(optimizer, model_configs, num_samples)
    elif sched_s == 'StepLR':
        return StepLR(optimizer, sched_dict['args']['step_size'])
    elif sched_s == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer)
    else:
        assert f'Could not find {sched_s} in schedulers dictionary'


class ECCLR(StepLR):

    def __init__(self, optimizer, model_configs):
        self.gamma = model_configs['scheduler']['args']['gamma']
        self.step_size = model_configs['scheduler']['args']['step_size']  # does not matter
        super(ECCLR, self).__init__(optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=-1)

    def get_lr(self):
        if self.last_epoch in [25, 35, 45]:
            return [group['lr'] * self.gamma
                    for group in self.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, optimizer, model_configs, num_samples):
        """
        Initializes the learning rate scheduler.
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        warmup_epochs = model_configs['scheduler']['args']['warmup_epochs']
        total_epochs = [model_configs['num_epochs']] * model_configs['num_lrs']
        steps_per_epoch = num_samples // model_configs['batch_size']
        init_lr = model_configs['scheduler']['args']['init_lr']
        max_lr = model_configs['scheduler']['args']['max_lr']
        final_lr = model_configs['scheduler']['args']['final_lr']

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]