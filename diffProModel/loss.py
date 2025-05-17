import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss function.
    
    Encourages 'anchor' to be close to 'positive' samples and far from 'negative' samples.
    """
    def __init__(self, margin=1.0):
        """Initializes ContrastiveLoss.

        Args:
            margin (float, optional): The margin for the loss. Defaults to 1.0.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """Computes the contrastive loss.

        Args:
            anchor (torch.Tensor): Embeddings of the anchor samples.
            positive (torch.Tensor): Embeddings of the positive samples.
            negative (torch.Tensor): Embeddings of the negative samples.

        Returns:
            torch.Tensor: The mean contrastive loss.
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        # Loss = max(0, pos_dist - neg_dist + margin)
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss
