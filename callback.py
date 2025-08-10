import torch
import torch.nn.functional as F


from pytorch_lightning import Callback

import matplotlib.pyplot as plt

plt.style.use('ggplot')

class LRP_callback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss_hard = []
        self.val_loss_hard = []
        self.train_acc_hard = []
        self.val_acc_hard = []

        self.sparsity = []

        self.tau = []
        self.r = []

        self.gradients = {"tau": [], "r": [], "weights": [], "weights_std": []}
        self.t = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss')
        acc = trainer.callback_metrics.get('train_acc')
        hard_loss = trainer.callback_metrics.get('train_loss_hard')
        hard_acc = trainer.callback_metrics.get('train_acc_hard')

        if loss is not None:
            self.train_loss.append(loss.item())
        if acc is not None:
            self.train_acc.append(acc.item())
        if hard_loss is not None:
            self.train_loss_hard.append(hard_loss.item())
        if hard_acc is not None:
            self.train_acc_hard.append(hard_acc.item())



    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        
        tau = trainer.callback_metrics.get('tau')
        r = trainer.callback_metrics.get('r')
        t = trainer.callback_metrics.get('t')

        
        if tau is not None:
            self.tau.append(tau.item())
        if r is not None:
            self.r.append(r.item())
        if t is not None:
            self.t.append(t.item())

        for param_name in ["tau", "r"]:
            param = getattr(pl_module, param_name, None)
            if param is not None and param.grad is not None:
                self.gradients[param_name].append(param.grad.item())  # Store scalar value
            else:
                self.gradients[param_name].append(None)

        weight_grads = []
        for name, param in pl_module.named_parameters():
            if "weight" in name and param.grad is not None:
                weight_grads.append(param.grad.view(-1))  # Flatten weight gradient tensor

        if weight_grads:
            all_weight_grads = torch.cat(weight_grads)  # Combine all weight gradients
            grad_mean = all_weight_grads.mean().item()
            grad_std = all_weight_grads.std().item()
            self.gradients["weights"].append(grad_mean)
            self.gradients["weights_std"].append(grad_std)
        else:
            self.gradients["weights"].append(None) 
            self.gradients["weights_std"].append(None)
    

    def on_validation_epoch_end(self, trainer, pl_module):
        
        loss = trainer.callback_metrics.get('val_loss')
        acc = trainer.callback_metrics.get('val_acc')
        hard_loss = trainer.callback_metrics.get('val_loss_hard')
        hard_acc = trainer.callback_metrics.get('val_acc_hard')

        sparsity = trainer.callback_metrics.get('sparsity')

        if loss is not None:
            self.val_loss.append(loss.item())
        if acc is not None:
            self.val_acc.append(acc.item())
        if hard_loss is not None:
            self.val_loss_hard.append(hard_loss.item())
        if hard_acc is not None:
            self.val_acc_hard.append(hard_acc.item())

        if sparsity is not None:
            self.sparsity.append(sparsity.item())

    def plot_metrics(self):
        # Plot training and validation loss
        plt.figure(figsize=(14, 6))

        # plt.suptitle("Training and Validation Loss and Accuracy (Soft Mask)", fontsize=16)
        # Loss plot
        # plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label="Train Loss", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.plot(range(1, len(self.val_loss[1:]) + 1), self.val_loss[1:], label="Val Loss", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("Loss", fontsize=25)
        # plt.title("Training and Validation Loss", fontsize=25)
        plt.legend()
        plt.show()

        # Accuracy plot
        # plt.subplot(1, 2, 2)
        plt.figure(figsize=(14, 6))
        plt.plot(range(1, len(self.train_acc) + 1), self.train_acc, label="Train Accuracy", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.plot(range(1, len(self.val_acc[1:]) + 1), self.val_acc[1:], label="Val Accuracy", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("Accuracy", fontsize=25)
        # plt.title("Training and Validation Accuracy", fontsize=25)
        plt.legend()

        # plt.tight_layout()
        plt.show()

        print(f'final train : {self.train_loss[-1]}, final val loss: {self.val_loss[-1]}')
        print(f'final train acc : {self.train_acc[-1]}, final val acc: {self.val_acc[-1]}')

        plt.figure(figsize=(14, 6))

        # plt.suptitle("Training and Validation Loss and Accuracy (Hard Mask)", fontsize=16)
        # Loss plot
        # plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_loss_hard) + 1), self.train_loss_hard, label="Train Loss", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.plot(range(1, len(self.val_loss_hard[1:]) + 1), self.val_loss_hard[1:], label="Val Loss", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("Loss", fontsize=25)
        # plt.title("Training and Validation Loss", fontsize=25)
        plt.legend()
        plt.show()

        # Accuracy plot
        # plt.subplot(1, 2, 2)
        plt.figure(figsize=(14, 6))
        plt.plot(range(1, len(self.train_acc_hard) + 1), self.train_acc_hard, label="Train Accuracy", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.plot(range(1, len(self.val_acc_hard[1:]) + 1), self.val_acc_hard[1:], label="Val Accuracy", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("Accuracy", fontsize=25)
        # plt.title("Training and Validation Accuracy", fontsize=25)
        plt.legend()

        # plt.tight_layout()
        plt.show()

        print(f'final train : {self.train_loss_hard[-1]}, final val loss: {self.val_loss_hard[-1]}')
        print(f'final train acc : {self.train_acc_hard[-1]}, final val acc: {self.val_acc_hard[-1]}')

        plt.figure(figsize=(14, 6))

        plt.plot(range(1, len(self.sparsity) + 1), self.sparsity, label="Sparsity", marker = 'o', linewidth = 3, alpha = 0.75)
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("Sparsity", fontsize=25)
        # plt.title("Sparsity", fontsize=25)
        plt.legend()
        plt.show()

        print(f'final sparsity: {self.sparsity[-1]}')

    def plot_parameters(self):
        plt.figure(figsize=(14, 6))

        # plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.tau) + 1), self.tau, label="Tau", marker = 'o', linewidth = 1, alpha = 0.75, color = 'blue')
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("Tau", fontsize=25)
        # plt.title("Tau", fontsize=25)
        plt.legend()

        # plt.subplot(1, 2, 2)
        plt.figure(figsize=(14, 6))
        plt.plot(range(1, len(self.r) + 1), self.r, label="r", marker = 'o', linewidth = 1, alpha = 0.75, color = 'blue')
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("r", fontsize=25)
        # plt.title("r", fontsize=25)
        plt.legend()

        # plt.tight_layout()
        plt.show()


    def plot_gradients(self, param_name="tau"):
    
        if param_name in ["tau", "r"]:
            data = self.gradients[param_name]
            steps = list(range(len(data)))

            plt.figure(figsize=(10, 5))
            plt.plot(steps, data, label=f"{param_name} Gradient", color="blue")

        elif param_name == "weights":
            data = self.gradients["weights"]
            std = self.gradients["weights_std"]
            means = [x for x in data if x is not None]
            stds = [x for x in std if x is not None]
            steps = list(range(len(means)))

            means = torch.tensor(means)
            stds = torch.tensor(stds)

            plt.figure(figsize=(10, 5))
            plt.plot(steps, means, label="Global Weight Gradient Mean", color="blue")
            plt.fill_between(steps, means - stds, means + stds, color="blue", alpha=0.3, label="±1 Std Dev")

        else:
            print(f"Parameter {param_name} not found in stored gradients.")
            return

        plt.xlabel("Training Steps")
        plt.ylabel("Gradient Value")
        plt.title(f"{param_name} Gradient Over Training")
        plt.legend()
        plt.grid(True)
        plt.show()

