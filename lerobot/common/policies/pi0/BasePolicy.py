# import os
# import json
# import torch
# import wandb
# import torch.nn as nn
#
# class BasePolicy(nn.Module):
#     def __init__(
#             self,
#             net,
#             loss_configs,
#             do_compile,
#         ):
#         super().__init__()
#         self.do_compile = do_compile
#         if do_compile:
#             self.net = torch.compile(net)
#         else:
#             self.net = net
#         self.use_ema = False
#         self.loss_configs = loss_configs
#
#         print("number of parameters: {:e}".format(
#             sum(p.numel() for p in self.net.parameters()))
#         )
#
#
#
#     def _compute_loss(self, batch):
#         pred = self.net(batch)
#         loss = {'total_loss': 0}
#         for key in self.loss_configs:
#             loss_func = self.loss_configs[key]['loss_func']
#             weight = self.loss_configs[key]['weight']
#             loss[key] = loss_func(pred[key], batch[key])
#             loss['total_loss'] += loss[key] * weight
#         return loss
#
#     def _infer(self, batch):
#         return self.net(batch)
