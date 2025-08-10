import torch
import torch.nn as nn
from torchmetrics import Accuracy
import torch.nn.functional as F

import pytorch_lightning as pl


import numpy as np
import matplotlib.pyplot as plt

from Custom_profiler import Custom_Profiler
from kde_fft import KDEFFT




class LRP(pl.LightningModule):
    def __init__(self, r0, tau0, tau0_std_scale: bool = True, lr_r0 = 1e-5, lr_tau0 = 1e-6, lamb = 1e-6, profiler: Custom_Profiler = None):
        super(LRP, self).__init__()

        self.automatic_optimization = False # Nao me lembro para o que isto é

        self.use_hard_mask = False  # Default: soft during training
        self.lr_r0 = lr_r0
        self.lr_tau0 = lr_tau0
        self.lamb = lamb

        self.sort_count = 1

        self.r = nn.Parameter(torch.tensor(r0, dtype=torch.float32, requires_grad=True))

        self.masks = [] # Store masks for each layer
        self.prunable_layers_idx = [] # Store prunable layers
        self.sorted_weights = []

        self.profiler = profiler
        
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        # self.layers = nn.Sequential(
        #     nn.Linear(28*28, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10)
        # )

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # if i == 0 or i == len(self.layers) - 1:
                    # continue
                self.prunable_layers_idx.append(i) # Extract prunable layers

        self._all_weights = torch.cat([self.layers[i].weight.view(-1) for i in self.prunable_layers_idx])

        if tau0_std_scale:
            self.tau = nn.Parameter(torch.tensor(tau0 * torch.std(self._all_weights), dtype=torch.float32, requires_grad=True))
        else:
            self.tau = nn.Parameter(torch.tensor(tau0, dtype=torch.float32, requires_grad=True))

        self._update_sorted_weights()

        self.t = self._compute_global_t()  # Compute global threshold t

        self._update_masks()  # Compute masks for each layer
        
        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

        if self.profiler:# this is made to assign a decorator to the functions that we want to profile, se usage example to better understand how to use it
            for func_name in self.profiler.functions_to_profile:
                if hasattr(self, func_name):
                    setattr(self, func_name, self.profiler.profile(getattr(self, func_name)))

    """
    Funções internas
    """
    @staticmethod
    def _safe_softmax(w, t, tau): # usando softmax sem esta normalizacao da overflow facilmente
        x1 = w**2 / tau
        x2 = t**2 / tau
        max_val = torch.maximum(x1, x2)  # prevent overflow
        exp1 = torch.exp(x1 - max_val)
        exp2 = torch.exp(x2 - max_val)
        return exp1 / (exp1 + exp2)

    @torch.no_grad()
    def _update_masks(self):
        """Computes masks for each layer based on current weights and global threshold t."""
        self.masks = [] # Reset masks

        for i in self.prunable_layers_idx:
            weight = self.layers[i].weight
            # soft_mask = torch.exp(weight**2 / self.tau) / (torch.exp(weight**2 / self.tau) + torch.exp(self.t**2 / self.tau))
            soft_mask = self._safe_softmax(weight, self.t, self.tau)  # Compute soft mask using the safe softmax function
            soft_mask = soft_mask.to(weight.device)  # Ensure mask is on the same device as weight
            self.masks.append(soft_mask)

    @torch.no_grad()
    def _update_sorted_weights(self):
        """Computes de absolute value of the weights and sorts them by magnitude"""

        new_abs_weights = torch.cat([torch.abs(self.layers[i].weight.view(-1)) for i in self.prunable_layers_idx])

        new_abs_weights = new_abs_weights.detach()

        if not hasattr(self, "prev_sorted_weights"):
            
            self.sorted_weights = torch.sort(new_abs_weights)[0]#torch.sort(new_abs_weights)#.detach().cpu().numpy()
        else:
        
            self.sorted_weights = self._incremental_sort(self.prev_sorted_weights, new_abs_weights)

        
        self.prev_sorted_weights = self.sorted_weights.clone()

    def _incremental_sort(self, prev_sorted_weights, new_abs_weights):
        """Performs an incremental update to the sorted order instead of full sorting."""

        # Make sure both are torch tensors on the same device
        if isinstance(prev_sorted_weights, np.ndarray):
            prev_sorted_weights = torch.from_numpy(prev_sorted_weights).float()
        prev_sorted_weights = prev_sorted_weights.to(new_abs_weights.device)

        # Compute absolute differences between new and previous sorted weights
        diffs = (new_abs_weights - prev_sorted_weights).abs()

        # Define threshold for significant changes (Mean + Std Dev)
        change_threshold = diffs.mean() + diffs.std()

        # Find indices of changed elements
        changed_indices = torch.where(diffs > change_threshold)[0]

        if len(changed_indices) > 0:
            # Only re-sort the changed weights
            self.sort_count += 1
            sorted_weights = torch.sort(new_abs_weights)[0]#torch.sort(new_abs_weights)#.detach().cpu().numpy()
            return sorted_weights
        else:
            # If changes are minimal, return the previous order
            return prev_sorted_weights

    @torch.no_grad()
    def _compute_global_t(self):
        """Computes a single global threshold t based on all network weights."""

        sorted_weights = self.sorted_weights
        
        k = int(self.r * len(sorted_weights))  # Number of elements to prune
        
        if k == 0 or k >= len(sorted_weights):
            return torch.tensor(0.0)  # No pruning if r=0


        t = 0.5 * (sorted_weights[k - 1] + sorted_weights[k])  # Midpoint between largest pruned and smallest kept
        
        return torch.tensor(t, device = self.device)

    @torch.no_grad()
    def _compute_tau_gradient(self):

        tau_grad = torch.tensor(0.0, device=self.tau.device)
        
        for mask, i in zip(self.masks, self.prunable_layers_idx):
            
            weight = self.layers[i].weight
            weight_grad = self.layers[i].weight.grad
            
            if weight_grad is None or mask is None:
                continue

            tau_grad_ = (weight * (self.t**2 - weight**2) / self.tau**2) * mask * (1 - mask) * weight_grad
            tau_grad += tau_grad_.sum()
            
        return tau_grad

    @torch.no_grad()
    def _compute_r_gradient(self):
        
        r_grad = torch.tensor(0, dtype = torch.float32, device = self.r.device)

        for mask, i in zip(self.masks, self.prunable_layers_idx):
            
            weight = self.layers[i].weight
            weight_grad = self.layers[i].weight.grad

            if weight_grad is None or mask is None:
                continue

            r_grad_ =  (- 2 * weight * self.t / self.tau) * mask * (1 - mask) * weight_grad

            r_grad_ = r_grad_.to(dtype=torch.float32)
            # print(r_grad_)
            # print(self._compute_dtdr())
            # print("correu")
            # print(r_grad.device, self._compute_dtdr().device)
            r_grad += r_grad_.sum().float()
        
        r_grad = r_grad * self._compute_dtdr()

        r_grad += self.lamb * 2 * (self.r - 1) #  regularization term

        return r_grad

    @torch.no_grad()
    def _compute_dtdr(self):
        sorted_weights = self.sorted_weights

        # Ensure it's a NumPy array
        if torch.is_tensor(sorted_weights):
            sorted_weights = sorted_weights.detach().cpu().numpy()

        n = len(sorted_weights)
        k = int(self.r * n)
        
        # Handle edge cases
        if k <= 0 or k >= n:
            return torch.tensor(0.0, dtype=torch.float32) 
    
        # kde = gaussian_kde(sorted_weights, bw_method = "scott")
        kde = KDEFFT(sorted_weights)
        
        dtdr = -0.5 * (1/kde(sorted_weights[k]) + 1/kde(sorted_weights[k - 1]))

        return torch.tensor(dtdr, dtype = torch.float32).squeeze(0)


    @torch.no_grad()
    def _update_weight_gradient(self):

        for mask, i in zip(self.masks, self.prunable_layers_idx):
            weight = self.layers[i].weight
            weight_grad = self.layers[i].weight.grad

            if weight_grad is None or mask is None:
                continue

            grad_update = mask * weight_grad + 2 * (weight**2 / self.tau) * mask * (1 - mask) * weight_grad

            self.layers[i].weight.grad = grad_update

    @torch.no_grad()
    def _apply_hard_pruning(self):
        """Applies hard pruning to the weights based on the threshold."""
        
        for mask, i in zip(self.masks, self.prunable_layers_idx):
            weight = self.layers[i].weight

            hard_mask = (mask >= 0.5).float().to(weight.device)  # Convert to binary mask
            # print(hard_mask)

            weight.data = weight.data * hard_mask  # Apply hard mask to weights


        
    """
    Funções principais
    """


    def forward(self, x):
        x = torch.flatten(x, 1)

        # print(f"Type of self.r: {type(self.r)}")
        # print(f"Type of self.tau: {type(self.tau)}")
        # print(f"Type of self.t: {type(self.t)}")
        # print(self.tau.device)
        # print(self.r.device)
        # print(self.t.device)

        mask_counter = 0
        for i, layer in enumerate(self.layers):
            if i in self.prunable_layers_idx:
                
                weight = self.layers[i].weight
                
                # mask = getattr(self, f"mask_{i}")
                mask = self.masks[mask_counter]
                mask = mask.to(weight.device)  # Ensure mask is on the same device as weight

                if self.use_hard_mask:
                    mask = (mask >= 0.5).float().to(weight.device)  # Convert to binary mask

                masked_weight = weight * mask
                
                x = torch.nn.functional.linear(x, masked_weight, layer.bias)
                

                mask_counter += 1

            else:
                x = layer(x)

        return x

    def training_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        loss = F.cross_entropy(output, target)
        acc = self.accuracy(output, target)

        masks = torch.cat([m.flatten() for m in self.masks]) # Differentiable L0 regularization on masks
        # regularization_loss = - self.lamb * torch.log(1 - masks + 1e-6).mean()
        regularization_loss = self.lamb * (self.r - 1) ** 2
        # regularization_loss = -self.lamb * self.r
        loss += regularization_loss

        self.use_hard_mask = True  # Temporarily switch
        output_hard = self(data)
        loss_hard = F.cross_entropy(output_hard, target)
        acc_hard = self.accuracy(output_hard, target)
        self.use_hard_mask = False  # Switch back

    
        opt1, opt2, opt3 = self.optimizers()

        opt1.zero_grad()
        opt2.zero_grad()
        opt3.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        opt2.step()
        opt3.step()

        with torch.no_grad():
            self.tau.data.clamp_(min=1e-6)
            self.r.data.clamp_(min=0, max=0.9999)  

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_loss_hard", loss_hard, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc_hard", acc_hard, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log("r", self.r.clone().detach().cpu().item(), on_step=True, on_epoch=True, prog_bar=True)
        # print(f"r: {self.r.detach().cpu().item()}")
        self.log("tau", self.tau.clone().detach().cpu().item(), on_step=True, on_epoch=True, prog_bar=True)
        # print(f"tau: {self.tau.detach().cpu().item()}")

        # Temporario
        self.log('t', self.t.clone().detach().cpu().item(), on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        loss = F.cross_entropy(output, target)
        acc = self.accuracy(output, target)

        self.use_hard_mask = True  # Temporarily switch
        output_hard = self(data)
        loss_hard = F.cross_entropy(output_hard, target)
        acc_hard = self.accuracy(output_hard, target)
        self.use_hard_mask = False  # Switch back

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        self.log("val_loss_hard", loss_hard, on_epoch=True, prog_bar=True)
        self.log("val_acc_hard", acc_hard, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            for mask in self.masks:
                hard_mask = (mask >= 0.5).float().to(data.device) 
                sparsity = 1 - torch.mean(hard_mask)
                
        self.log("sparsity", sparsity, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        acc = self.accuracy(output, target)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

    #@torch.no_grad()
    def on_before_optimizer_step(self, optimizer):

        self._update_masks()
        
        if self.tau.grad is None:
            self.tau.grad = torch.zeros_like(self.tau)
        if self.r.grad is None:
            self.r.grad = torch.zeros_like(self.r)

        self.tau.grad = self._compute_tau_gradient()
        self.r.grad = self._compute_r_gradient()

        self._update_weight_gradient()

    def on_after_backward(self):

        self._update_sorted_weights()
        self.t = self._compute_global_t()
        

    def on_train_end(self):
        """Automatically print profiler summary at the end of training."""
        if self.profiler:
            self.profiler.end()

    def on_fit_end(self):
        self._apply_hard_pruning()  # Apply hard pruning at the end of training
        self.use_hard_mask = True  # Use hard mask for validation and testing

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters())

        layer_params = [p for p in self.layers.parameters() if p.requires_grad]
        optimizer1 = torch.optim.Adam(layer_params, lr=1e-3)

        optimizer2 = torch.optim.Adam([self.r], lr=self.lr_r0)

        optimizer3 = torch.optim.Adam([self.tau], lr=self.lr_tau0)

        return [optimizer1, optimizer2, optimizer3]

    #     optimizer = torch.optim.Adam([
    #     {'params': [p for p in self.layers.parameters() if p.requires_grad], 'lr': 1e-3},
    #     {'params': [self.r], 'lr': 1e-4},
    #     {'params': [self.tau], 'lr': 1e-5},
    # ])
    #     return optimizer


    """
    Funções de vizualização
    """

    def viz(self, tolerance_zeros = 0, full_viz = False):
        zero_proportions = []

        for i, layer in enumerate(self.layers):

            if isinstance(layer, nn.Linear):# saltar as camadas ReLU


                weights = layer.weight.data.cpu().numpy()

                masked_array = np.ma.masked_inside(weights, -tolerance_zeros, tolerance_zeros)# Aqui masked array é outra coisa
                cmap = plt.get_cmap('viridis')                                                # não tem nada a ver com a máscara
                cmap.set_bad(color='white')

                zero_proportions.append(
                    np.mean(masked_array.mask) * 100
                )

                if (i == 0 or i == len(self.layers) - 1) and full_viz:
                    fig, axs = plt.subplots(ncols = 2, figsize = (18, 6))
                    fig.suptitle(f'layer {i}')
                    cax = axs[0].imshow(masked_array, cmap = cmap)
                    fig.colorbar(cax)
                    axs[0].grid(False)
                    axs[1].hist(weights.ravel(),
                                bins = 50,
                                color = 'skyblue',
                                edgecolor = 'white',
                                alpha = 0.65
                                )
                    plt.show()
                elif full_viz:
                    fig, axs = plt.subplots(ncols = 2, figsize = (18, 6))
                    #fig.suptitle(f'layer {i//2}')
                    cax = axs[0].imshow(masked_array, cmap = cmap)
                    fig.colorbar(cax)
                    axs[0].grid(False)
                    axs[1].hist(weights.ravel(),
                                bins = 64,
                                color = 'skyblue',
                                edgecolor = 'white',
                                alpha = 0.65
                                )

                    axs[1].set_xlabel('X', size = 18)
                    axs[1].set_ylabel('Count', size = 18)
                    plt.legend(loc = 1, fontsize = 14)
                    plt.show()

        plt.figure(figsize = (18, 6))
        plt.plot(range(1, len(zero_proportions) + 1), zero_proportions, '--o')
        plt.title('zero proportions over layers')
        plt.xlabel('layer')
        plt.ylabel('zero proportion')
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim([0, 100])

        plt.show()

    def viz_masks(self):
        for i, mask in enumerate(self.masks):
            plt.figure(figsize=(8, 6))
            print(mask)
            # plt.subplot(1, 2, 1)
            plt.imshow(mask.cpu().numpy(), cmap='viridis')
            # plt.title(f"Mask for layer {i + 1}")
            plt.grid(False)
            plt.colorbar()
            plt.show()
            

            # plt.subplot(1, 2, 2)
            plt.figure(figsize=(14, 6))

            plt.hist(mask.cpu().numpy().ravel(), bins=50, color='skyblue', edgecolor='white', alpha=0.65)
            plt.xlabel('Mask Value')
            plt.ylabel('Count')
            # plt.title(f"Mask Histogram for layer {i + 1}")
            plt.grid(True)
            
            plt.show()

    @torch.no_grad()
    def viz_softmax(self, definition = 1024):
        tau = self.tau.clone().detach().cpu().numpy()
        t = self.t.clone().detach().cpu().numpy()
        
        softmax = lambda w: 0.5 * (1 + np.tanh((w**2 - t**2) / (2 * tau)))

        for mask, i in zip(self.masks, self.prunable_layers_idx):
            weight = self.layers[i].weight
            weight = weight.clone().detach().cpu().numpy()
            weight = abs(weight)

            min_w, max_w = np.min(weight), np.max(weight)

        #     weight = self.layers[i].weight.detach()
        #     weight = abs(weight)
            
        #     soft_mask = torch.exp(weight**2 / self.tau) / (torch.exp(weight**2 / self.tau) + torch.exp(self.t**2 / self.tau))
        ws = np.linspace(min_w, max_w, definition)

        plt.figure(figsize = (18, 6))

        # plt.scatter(weight.cpu().numpy(), soft_mask.cpu().numpy(), label = fr'$\tau$ = {self.tau}', color = 'skyblue', alpha = 0.65, edgecolors = 'blue')
        plt.plot(ws, softmax(ws), label = fr'$\tau$ = {self.tau}', color = 'skyblue', alpha = 0.65, linewidth = 3)
        plt.xlabel('x')
        plt.ylabel(r'softmax(x, $\tau$)')
        plt.legend()
        epsilon = 0.1
        plt.ylim([0 - epsilon, 1 + epsilon])

        plt.show()