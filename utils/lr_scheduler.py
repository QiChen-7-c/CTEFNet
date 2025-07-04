import numpy as np

class LRwarmup(object):
    def __init__(self, optimizer, lr_max=0.001, lr_min=0.0001, warm_milestone=10, annealing_index=0.5):
        self.optimizer = optimizer
        self.step_cnt = 0
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warm_milestone = warm_milestone
        self.annealing_index = annealing_index

    def step(self, step=None):
        if step:
            self.step_cnt = step
        self.step_cnt += 1
        lr_val = self.lr_max * min((self.warm_milestone/self.step_cnt)**(self.annealing_index), self.step_cnt * self.warm_milestone ** (-1.))
        if lr_val < self.lr_min:
            lr_val = self.lr_min
        return self.__set_lr(lr_val)

    def __set_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            # print('lr_scheduler', i, len(param_group['params']))
            if len(param_group['params']) == 1:
                if self.step_cnt < 3000:
                    param_group['lr'] = 0.002
                else:
                    param_group['lr'] = 0
            else:
                param_group['lr'] = lr
        return self.get_lr() == lr  # check whether to set correctly

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


if __name__=='__main__':
    import torch
    from network.ctefnet import CTEFNet

    model = CTEFNet(input_size=(120, 180), dim=256, head=4, in_channels=8, obs_time=12, pred_time=24, dim_feedforward = 512, num_index=1)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    params1 = [p for n, p in model.named_parameters() if 'input_reg' in n]
    params2 = [p for n, p in model.named_parameters() if 'input_reg' not in n]
    optimizer = torch.optim.Adam([
        {"params": params1},
        {"params": params2},
    ], lr=0.001, weight_decay=0)

    lr_scheduler = LRwarmup(
        optimizer,
        lr_max=0.0002,
        lr_min=0.000001,
        warm_milestone=1000,
        annealing_index=0.2,
    )

    lr_scheduler.step()
    # lr_scheduler.step()

    # for param_group in optimizer.state_dict()['param_groups']:
    #     print(param_group['lr'], param_group['lr'])
