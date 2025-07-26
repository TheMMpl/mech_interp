import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformer_lens import HookedTransformer, utils

from consts import DTYPES

class AutoEncoder(nn.Module):
    """
    Simple autoencoder for feature extraction, inspired by Neel Nanda's notebook.
    """
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_emb"] * cfg["dict_mult"]
        d_mlp = cfg["d_emb"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        # self.to("cuda")

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        Args:
            x: Input tensor
        Returns:
            x_reconstruct: Reconstructed input
            acts: Feature activations
        """
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct, acts


class SparseTranscoder(pl.LightningModule):
    """
    PyTorch Lightning module for sparse autoencoding of transformer activations.
    """
    def __init__(self, config, hooked_transformer, layer, **kwargs):
        super().__init__(**kwargs)
        self.transcoder = AutoEncoder(cfg=config)
        self.llm = hooked_transformer
        self.layer = layer
        self.lr = config['lr']
        # for param in self.transcoder.parameters():
        #     print(param)
        # self.llm.eval()

    def forward(self, x):
        """
        Forward pass through the transcoder.
        Args:
            x: Input tensor
        Returns:
            x_reconstruct, acts: Output and activations
        """
        return self.transcoder(x)

    # for now we sample the llm during training - in the future maybe caching activations will prove usefull
    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        """
        with torch.no_grad():
            logits, model_cache = self.llm.run_with_cache(batch['input_ids'])
        if type(self.layer) == int:
            input_acts = model_cache[f'blocks.{self.layer}.ln2.hook_normalized']
            output_acts = model_cache[f'blocks.{self.layer}.hook_mlp_out']
        else:
            raise NotImplementedError()

        preds, acts = self.forward(input_acts)
        l2_loss = (preds.float() - output_acts.float()).pow(2).sum(-1).mean(0).sum()
        l1_loss = self.transcoder.l1_coeff * (acts.float().abs().sum()).sum()
        loss = l2_loss + l1_loss
        self.log('train_loss', loss)
        self.log('reconstruction_loss', l2_loss)
        self.log('sparsity_loss', l1_loss)
        return loss

    # for now we sample the llm during training - in the future maybe caching activations will prove usefull
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        """
        with torch.no_grad():
            # what's up with it leaving as many outputs as logits, and not patch dim?
            # likely it's doing tokenwise predictions - to check - so 50 predictions per fragment
            logits, model_cache = self.llm.run_with_cache(batch['input_ids'])
        if type(self.layer) == int:
            input_acts = model_cache[f'blocks.{self.layer}.ln2.hook_normalized']
            output_acts = model_cache[f'blocks.{self.layer}.hook_mlp_out']
        else:
            raise NotImplementedError()
        preds, acts = self.forward(input_acts)
        l2_loss = (preds.float() - output_acts.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.transcoder.l1_coeff * (acts.float().abs().sum())
        # the optimizer doesn't sum this automatically -  so this is equivalent, right?
        loss = (l2_loss + l1_loss).sum()
        self.log('val_loss', loss)
        # self.log('reconstruction_loss', l2_loss)
        # self.log('sparsity_loss',l1_loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        return torch.optim.Adam(self.transcoder.parameters(), lr=self.lr)