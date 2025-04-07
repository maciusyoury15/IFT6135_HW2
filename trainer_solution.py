import torch
# from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm
import os
from collections import defaultdict
# from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
# import time

########################################################################################
########################################################################################

def get_loss_and_accuracy(logits, targets, eq_positions, mask, reduction='mean'):
    """
    Computes the mean negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.

    The equation can be : 
        - "[BOS] [a] [+] [b] [=] [r] [EOS] [PAD] [PAD]", in that case target is "[a] [+] [b] [=] [r] [EOS] [PAD] [PAD]"
        - "[BOS] [a] [+] [b] [+] [c] [=] [r] [EOS]", in that case target is "[a] [+] [b] [+] [c] [=] [r] [EOS]"

    Let :
        - B : batch size
        - S : sequence length
        - V : vocabulary size
    
    Parameters
    ----------
    logits : torch.FloatTensor of shape (B, S, V)
        A tensor containing the logits of the next token for all positions in each sequence of the mini-batch.
    targets : torch.LongTensor of shape (B, S)
        A tensor containing the target next tokens for all positions in each sequence of the mini-batch.
    eq_positions : torch.LongTensor of shape (B,)
        The position of the '=' token in each sequence (each sample has exactly one '=').
    mask : torch.LongTensor of shape (B, S)
        A mask indicating valid tokens (1 if valid, 0 for PAD tokens).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        - 'none': no reduction will be applied
        - 'mean': average the output of the batch dimension. 
        - 'sum': sum the output of the batch dimension.
        
    Returns
    -------
    loss : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The negative log-likelihood loss computed over the valid (non-PAD) RHS tokens.
    accuracy : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The accuracy over the batch where a sequence is counted as correct only if 
        all valid RHS tokens are predicted correctly.
    """
    """
    Computes the mean negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.
    """
    # Convert logits to log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Extract RHS positions: everything after the '=' token
    B, S = targets.shape
    rhs_mask = torch.zeros_like(mask, dtype=torch.bool)
    for b in range(B):
        rhs_mask[b, eq_positions[b] + 1:] = True  # RHS starts right after '='

    # Combine RHS mask with PAD mask (ensure valid tokens only)
    valid_rhs_mask = rhs_mask & mask

    # Flatten the tensors for loss computation
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))  # Shape: (B*S, V)
    targets_flat = targets.view(-1)  # Shape: (B*S,)

    # Compute negative log likelihood loss, ignoring invalid positions
    loss = F.nll_loss(
        log_probs_flat,
        targets_flat,
        reduction='none',
        ignore_index=-100  # Use -100 as ignore index to avoid invalid tokens
    )

    # Reshape loss to match (B, S)
    loss = loss.view(B, S)
    loss = loss * valid_rhs_mask  # Apply mask to consider only RHS valid tokens

    # Average loss per sample in the batch
    #loss_per_sample = loss.sum(dim=1) / valid_rhs_mask.sum(dim=1).clamp(min=1)
    loss_per_sample = loss.sum(dim=1) / valid_rhs_mask.sum(dim=1)

    # Apply reduction
    if reduction == 'mean':
        loss = loss_per_sample.mean()
    elif reduction == 'sum':
        loss = loss_per_sample.sum()
    elif reduction == 'none':
        loss = loss_per_sample
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

    # Compute accuracy: RHS is correct if all tokens match
    predictions = log_probs.argmax(dim=-1)  # Get the predicted tokens
    correct = (predictions == targets) & valid_rhs_mask  # Valid & correct predictions

    # For accuracy, check all RHS tokens per sequence
    seq_correct = correct.sum(dim=1) == valid_rhs_mask.sum(dim=1)  # Correct sequence-wise
    accuracy_per_sample = seq_correct.float()  # Convert bool to float for averaging

    # Apply reduction for accuracy
    if reduction == 'mean':
        accuracy = accuracy_per_sample.mean()
    elif reduction == 'sum':
        accuracy = accuracy_per_sample.sum()
    elif reduction == 'none':
        accuracy = accuracy_per_sample
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

    return loss, accuracy

########################################################################################
########################################################################################


@torch.no_grad()
def eval_model(model, loader, device) :
    model.eval()
    acc = 0
    loss = 0
    n = 0
    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        mask = mask.to(device)
        logits, *_ = model(batch_x) # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        # batch_losses, batch_accuracies = get_loss_and_accuracy(logits, batch_y, eq_positions, mask, 'none')

        n += batch_x.shape[0]
        loss += batch_loss.item() * batch_x.shape[0]
        acc += batch_acc * batch_x.shape[0]

    #l2_norm = 0
    l2_norm = torch.sqrt(sum(torch.sum(p**2) for p in model.parameters())).cpu().item()

    ##########
    # You can add more metrics in the dictionary (e.g., l2 norm of the parameters, etc.) 
    ##########

    #return {"loss" : loss / n, "accuracy": acc / n}
    # on ajoute .cpu() parce que les valeurs doivent pouvoir etre converties en numpy pour etre visualisée avec ax.plot
    # or, les tensor sur cuda ne sont pas convertibles en numpy
    return { "loss" : loss / n, "accuracy": acc.cpu() / n, "l2_norm": l2_norm }

'''
@torch.no_grad()
def eval_model(model, loader, device) :
    model.eval()
    acc = 0
    loss = 0
    n = 0

    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        mask = mask.to(device)

        logits, *_ = model(batch_x) # (B, S, V)
        #batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        batch_losses, batch_accuracies = get_loss_and_accuracy(logits, batch_y, eq_positions, mask, 'none')

        B = batch_x.shape[0]

        operator_mask = torch.zeros(B, dtype=torch.bool).to(device)
        for b in range(B):
            if eq_positions[b] == 3:
                operator_mask[b] = True  # RHS starts right after '='
                n += 1
        #print("operator_mask")
        #print(operator_mask)
        # n += batch_x.shape[0]

        loss += torch.sum(batch_losses * operator_mask)
        acc += torch.sum(batch_accuracies * operator_mask)

    if n == 0:
        n = 1
    ##########
    # You can add more metrics in the dictionary (e.g., l2 norm of the parameters, etc.)
    ##########

    #return {"loss" : loss / n, "accuracy": acc / n}
    # on ajoute .cpu() parce que les valeurs doivent pouvoir etre converties en numpy pour etre visualisée avec ax.plot
    # or, les tensor sur cuda ne sont pas convertibles en numpy
    return {"loss" : loss.cpu() / n, "accuracy": acc.cpu() / n}
'''

########################################################################################
########################################################################################


def train(
    model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device, 
    exp_name:str, checkpoint_path:str,
    n_steps:int, eval_first:int=0, eval_period:int=1, print_step:int=1, save_model_step:int=1,  save_statistic_step:int=1,  
    verbose=True,
    ):
    """
    model (nn.Module) : The model to train
    train_loader (DataLoader) : Training data loader
    train_loader_for_eval (DataLoader) : Training data loader (for evaluation)
    test_loader (DataLoader) : Test/Val data loader
    optimizer (Optimizer) : Optimizer
    device (str) : Device (cpu, cuda, cuda:0, etc)
    exp_name (str) : experiment name
    checkpoint_path (str) : Path to save the model checkpoints ("/path/to/experiment")
    n_steps (int) : Number of training steps
    eval_first (int) : Number of consecutive evaluation step at the beginning of training
    eval_period (int) : Evaluation frequency
    print_step (int) : Print frequency
    save_model_step (int) : Step interval to save model checkpoints
    save_statistic_step (int) : Step interval to save statistics (train/test loss, accuracy, etc.)
    verbose (bool) : Verbosity of the training
    """

    ##############
    # Checkpoint path
    os.makedirs(checkpoint_path, exist_ok=True)

    ##############
    # Number of training epochs
    total_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    n_steps = total_epochs * len(train_loader)
    
    if verbose :
        print(f"Number of training epochs & steps: {total_epochs} {n_steps}")

    ##############

    all_metrics = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["train"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["test"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["steps_epoch"] = {}

    ##############

    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() :
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device) 
    for k, v in test_statistics.items() :
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(0)
    all_metrics["steps_epoch"][0] = 0


    ######################
    # Save model
    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")

    
    ##############

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    if verbose :
        to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
        to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
        to_print += f" | lr = {current_lr}"
        print(to_print)

    ##############

    cur_step = 1 
    tol_step = 0

    for epoch in tqdm(range(1, total_epochs+1), desc="Training", total=total_epochs):

        # start_time = time.time()
        
        for i, batch in enumerate(train_loader) :
            batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x) # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ==========================
            # TODO: Write your code here
            # ==========================
            # scheduler.step()
            # current_lr = scheduler.optimizer.param_groups[0]["lr"]
            # ==========================
            # ==========================
              
            if cur_step in [1, n_steps] or cur_step % eval_period == 0 or cur_step <= eval_first:
                train_statistics = eval_model(model, train_loader_for_eval, device)
                for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, device)
                for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step)
                all_metrics["steps_epoch"][cur_step] = epoch

            
            if  verbose and (cur_step in [1, n_steps] or cur_step%print_step==0) :
                to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
                to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
                to_print += f" | lr = {current_lr}"
                print(to_print)

            if cur_step in [1, n_steps] or cur_step%save_model_step==0 or cur_step <= eval_first : 
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
                

            if cur_step in [1, n_steps] or cur_step%save_statistic_step==0:
                #to_save = {k:v for k, v in all_metrics.items()}
                to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
                torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

            cur_step += 1

        # ==========================
        # TODO: Write your code here
        # ==========================
        ###
        # scheduler.step() 
        # current_lr = scheduler.optimizer.param_groups[0]["lr"]
        # ==========================
        # ==========================

        ##############
        # You can implement early stopping here.
        # That is, if the model does not improve for a certain number of steps, you can stop the training.
        ##############

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time for one step : {elapsed_time} seconds")

    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
    
    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(cur_step)
    all_metrics["steps_epoch"][cur_step] = epoch

    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

    return all_metrics
