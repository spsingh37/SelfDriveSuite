from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_dataloader: Dataset,
        validation_dataloader: None,
        lr_scheduler: None,
        epochs: int = 100,
        epoch: int = 0,
        notebook: bool = False,
        checkpoint_dir: str = 'experiment',
        save_frequency: int = 1
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.data = []
        self.checkpoint_dir = f"checkpoints/{self.checkpoint_dir}_{time.time()}"
        self.best_loss = 10.00
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter
            print("epoch: ", self.epoch)
            """Training block"""
            self._train()

            # if self.epoch % self.save_frequency == 0:
            #     model_name = 'epoch_' + str(self.epoch)+'.pth'
            #     torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, model_name))

            """Validation block"""
            if self.validation_dataloader is not None:
                loss_value = self._validate()

            if loss_value < self.best_loss:
                model_name = 'epoch_best.pth'
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, model_name))
                print(f"weights saved at epoch: {self.epoch}")
                self.best_loss = loss_value

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if (
                    self.validation_dataloader is not None
                    and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau"
                ):
                    self.lr_scheduler.step(
                        self.validation_loss[i]
                    )  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.step()

                    #self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate, self.data

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        torch.manual_seed(0)
        batch_iter = tqdm(
            enumerate(self.training_dataloader),
            "Training",
            total=len(self.training_dataloader),
            leave=False,
            disable=True,
        )
        torch.manual_seed(0)
        for i, (x, y) in batch_iter:
            input_x, target_y = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            # print("input_x shape: ", input_x.shape)
            out = self.model(input_x)  # one forward pass
            # print("train out.shape: ", out.shape)
            out = np.squeeze(out, axis=1)
            # print("loss computation")
            # print("input_x.shape: ", input_x.shape)
            # print("target_y.shape: ", target_y.shape)
            self.data = [input_x, target_y, out]
            # print("train out.shape: ", out.shape)
            loss = self.criterion(out, target_y)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            print("train loss_value: ", loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(
                f"Training: (loss {loss_value:.4f})"
            )  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.validation_dataloader),
            "Validation",
            total=len(self.validation_dataloader),
            leave=False,
            disable=True,
        )

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                # print(out.shape)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                
                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")

        self.validation_loss.append(np.mean(valid_losses))
        print("val loss_value: ", np.mean(valid_losses))
        batch_iter.close()
        return loss_value
