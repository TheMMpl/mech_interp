from transformer_lens import HookedTransformer
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from consts import DEVICE,DTYPES,CONFIG,MODEL_PATH,BATCH_SIZE, MAX_EPOCHS, LOG_STEPS
from model import SparseTranscoder
from utils import prepare_training_data


llm = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    default_padding_side='left',
)

model=SparseTranscoder(CONFIG,llm,0)

train_loader,val_loader=prepare_training_data()
wandb_logger = WandbLogger(project="transcoders",log_model="all")
checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=10,every_n_epochs=1)
trainer = Trainer(logger=wandb_logger,callbacks=[checkpoint_callback],max_epochs=MAX_EPOCHS,log_every_n_steps=LOG_STEPS)
trainer.fit(model,train_loader,val_loader)