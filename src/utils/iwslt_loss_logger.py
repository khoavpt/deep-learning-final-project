import pytorch_lightning as pl
import matplotlib.pyplot as plt


class LossLogger(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []


    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        self.train_losses.append(train_loss.item())

#         if trainer.current_epoch % 10 == 0:
#             print(f"Train Loss: {train_loss.item()}")


    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        
        self.val_losses.append(val_loss.item())
#         if trainer.current_epoch % 10 == 0:
#             print(f"---- Epoch {trainer.current_epoch} ----")
#             print(f"Val Loss: {val_loss.item()}\t Val Pearson corr: {val_pearson_corr.item()}\t Val Spearman corr: {val_spearman_corr.item()}")
        
    
    def plot_losses(self):
        if self.train_losses and self.val_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot training and validation loss
            ax1.plot(self.train_losses, label='Train Loss')
            ax1.plot(self.val_losses, label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.set_title('Training and Validation Loss Over Epochs')
            plt.show()
        else:
            print("No data to plot.")
        
