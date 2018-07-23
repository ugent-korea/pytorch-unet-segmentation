import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.nlllos = nn.NLLLoss()
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1,2)
        y_true = y_true.view(-1)
        score = self.nlllos(self.logsoftmax(y_pred), y_true)
        return score
