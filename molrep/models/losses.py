import torch
from torch import nn
import numpy as np

from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

def get_loss_func(task_type, model_name, **kwargs):
    """
    Gets the loss function corresponding to a given dataset type.
    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    # if task_type == 'Classification' and model_name == 'DiffPool':
    #     return DiffPoolBinaryclassClassificationLoss()

    # if task_type == 'Regression' and model_name == 'DiffPool':
    #     return DiffPoolRegressionLoss()

    # if task_type == 'Classification' and model_name == 'VAE':
    #     return VAEClassificationLoss()

    # if task_type == 'Regression' and model_name == 'VAE':
    #     return VAERegressionLoss()

    if task_type == 'Classification':
        return BCEWithLogitsLoss(**kwargs)
        # return CrossEntropyLoss()

    if task_type == 'Regression':
        return MSELoss(**kwargs)
        # return MSERegressionLoss()

    if task_type == 'MultiClass-Classification':
        return CrossEntropyLoss(**kwargs)

    raise ValueError(f'Task type "{task_type}" and Model "{model_name}" not supported.')


# class ClassificationLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss = None

# class DiffPoolBinaryclassClassificationLoss(ClassificationLoss):
#     """
#     DiffPool - No Link Prediction Loss
#     """
#     def __init__(self, reduction=None):
#         super().__init__()
#         if reduction is not None:
#             self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
#         else:
#             self.loss = nn.BCEWithLogitsLoss()

#     def forward(self, targets, *outputs):
#         preds, lp_loss, ent_loss = outputs
#         preds, lp_loss, ent_loss = preds.to('cpu'), lp_loss.to('cpu'), ent_loss.to('cpu')

#         if targets.dim() > 1 and targets.size(1) == 1:
#             targets = targets.squeeze(1)

#         if preds.dim() > 1 and preds.size(1) == 1:
#             preds = preds.squeeze(1)

#         loss = self.loss(preds, targets.float())
#         return loss + lp_loss + ent_loss

# class CrossEntropyClassificationLoss(ClassificationLoss):
#     def __init__(self, reduction=None):
#         super().__init__()
#         if reduction is not None:
#             self.loss = nn.CrossEntropyLoss(reduction=reduction)
#         else:
#             self.loss = nn.CrossEntropyLoss()

# class VAEClassificationLoss(ClassificationLoss):
#     """
#     """
#     def __init__(self, reduction=None):
#         super().__init__()
#         if reduction is not None:
#             self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
#         else:
#             self.loss = nn.BCEWithLogitsLoss()

#     def forward(self, targets, *outputs):
#         preds, kl_loss, recon_loss = outputs
#         preds, kl_loss, recon_loss = preds.to('cpu'), kl_loss.to('cpu'), recon_loss.to('cpu')

#         if targets.dim() > 1 and targets.size(1) == 1:
#             targets = targets.squeeze(1)

#         if preds.dim() > 1 and preds.size(1) == 1:
#             preds = preds.squeeze(1)

#         loss = self.loss(preds, targets.float())
#         return loss + kl_loss + recon_loss



# class RegressionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss = None

#     def forward(self, targets, *outputs):
#         """
#         Args:
#             targets:
#             outputs:
        
#         return:
#             a loss value
#         """
#         raise NotImplementedError()


# # class MSERegressionLoss(RegressionLoss):
# #     def __init__(self, reduction=None):
# #         super().__init__()
# #         self.loss = nn.MSELoss(reduction=reduction)

# #     def forward(self, targets, *outputs):
# #         """
# #         Args:
# #             targets:
# #             outputs:
        
# #         return: 
# #             loss and accuracy values
# #         """
# #         outputs = outputs[0]
# #         if not isinstance(outputs, np.ndarray) and outputs.is_cuda:
# #             outputs = outputs.to('cpu')
# #         if len(outputs.size()) != 1 and len(targets.size()) == 1:
# #             outputs = outputs.reshape(-1)
# #         loss = self.loss(outputs, targets.float())
# #         return loss


# class DiffPoolRegressionLoss(RegressionLoss):
#     """
#     DiffPool - No Link Prediction Loss
#     """
#     def __init__(self, reduction=None):
#         super().__init__()
#         if reduction is not None:
#             self.loss = nn.MSELoss(reduction=reduction)
#         else:
#             self.loss = nn.MSELoss()

#     def forward(self, targets, *outputs):
#         preds, lp_loss, ent_loss = outputs
#         preds, lp_loss, ent_loss = preds.reshape(-1).to('cpu'), lp_loss.to('cpu'), ent_loss.to('cpu')

#         if targets.dim() > 1 and targets.size(1) == 1:
#             targets = targets.squeeze(1)

#         loss = self.loss(preds, targets.float())
#         return loss + lp_loss + ent_loss

# class VAERegressionLoss(RegressionLoss):
#     """
#     """
#     def __init__(self, reduction=None):
#         super().__init__()
#         if reduction is not None:
#             self.loss = nn.MSELoss(reduction=reduction)
#         else:
#             self.loss = nn.MSELoss()

#     def forward(self, targets, *outputs):
#         preds, kl_loss, recon_loss = outputs
#         preds, kl_loss, recon_loss = preds.to('cpu'), kl_loss.to('cpu'), recon_loss.to('cpu')

#         if targets.dim() > 1 and targets.size(1) == 1:
#             targets = targets.squeeze(1)

#         loss = self.loss(preds, targets.float())
#         return loss + kl_loss + recon_loss
