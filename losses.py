import torch
import torch.nn as nn


class Uncertainty(nn.Module):

    def __init__(self):
        # super(Uncertainty, self).__init__()
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros((3)))


        # self.log_vars = nn.Parameter(torch.FloatTensor([-0.33, -0.33, -0.33]))

    def forward(self, preds, emo, cou, age):
        emo_mse, cou_cross, age_mae = nn.MSELoss(), nn.CrossEntropyLoss(), nn.L1Loss()
        loss_emo = emo_mse(preds[0], emo)
        loss_cou = cou_cross(preds[1], cou)
        loss_age = age_mae(preds[2], age)

        loss_emo = loss_emo * torch.exp(-self.log_vars[0]) + self.log_vars[0]
        loss_cou = loss_cou * torch.exp(-self.log_vars[1]) + self.log_vars[1]
        loss_age = loss_age * torch.exp(-self.log_vars[2]) + self.log_vars[2]

        return loss_emo + loss_cou + loss_age


class UncertaintyRevised(nn.Module):

    def __init__(self):
        # super(Uncertainty, self).__init__()
        super().__init__()
        self.log_vars = nn.Parameter(torch.FloatTensor([-0.33, -0.33, -0.33]))

        # self.log_vars = nn.Parameter(torch.FloatTensor([-0.33, -0.33, -0.33]))

    def forward(self, preds, emo, cou, age):
        emo_mse, cou_cross, age_mae = nn.MSELoss(), nn.CrossEntropyLoss(), nn.L1Loss()
        loss_emo = emo_mse(preds[0], emo)
        loss_cou = cou_cross(preds[1], cou)
        loss_age = age_mae(preds[2], age)

        loss_emo = 0.33 / (self.log_vars[0] ** 2) * loss_emo + torch.log(1 + self.log_vars[0] ** 2)
        loss_cou = 0.33 / (self.log_vars[1] ** 2) * loss_cou + torch.log(1 + self.log_vars[1] ** 2)
        loss_age = 0.33 / (self.log_vars[2] ** 2) * loss_age + torch.log(1 + self.log_vars[2] ** 2)

        return loss_emo + loss_cou + loss_age