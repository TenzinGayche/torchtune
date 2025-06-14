from torchtune.models.gemma.rms_norm import GemmaRMSNorm
import torch.nn as nn


class GemmaMultiModalProjector(nn.Module):
    """
    The multi-modal projector for Gemma3. This module is responsible for
    projecting and normalizing the vision embeddings after they have been
    pooled by the vision encoder.
    """

    def __init__(
        self,
        mm_input_projection: nn.Module,
        mm_soft_emb_norm: nn.Module,

    ) -> None:
        super().__init__()
        self.mm_input_projection = mm_input_projection
        self.mm_soft_emb_norm = mm_soft_emb_norm

    def forward(self, x):
        # input is (batch_size, num_patches, input_dim)
        # The vision tower has already performed the average pooling.
        x = self.mm_input_projection(x)
        x = self.mm_soft_emb_norm(x)
        return x