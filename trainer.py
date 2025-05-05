import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau # No longer used
from torch.optim.lr_scheduler import LambdaLR # Import LambdaLR
import time
import wandb
import os
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import random
import math # Import math

# Assume util.py exists with these functions:
from util import save_checkpoint, load_checkpoint

# --- Learning Rate Scheduler Function ---
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Creates a learning rate scheduler with linear warmup and linear decay.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    # Ensure optimizer is passed correctly
    return LambdaLR(optimizer, lr_lambda, last_epoch)
# ---

class Trainer:
    def __init__(self, model, train_loader, val_loader, src_vocab, tgt_vocab,
                 config, device, checkpoint_dir='checkpoints'):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The Transformer model.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation/test set.
            src_vocab (Vocabulary): Source language vocabulary.
            tgt_vocab (Vocabulary): Target language vocabulary.
            config (dict): Configuration dictionary containing hyperparameters.
            device (torch.device): Device to run training on (e.g., 'cuda', 'cpu').
            checkpoint_dir (str): Directory to save checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader # This will be validation loader in train mode, test loader in test mode
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # --- Optimizer ---
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'], # Base learning rate
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # --- Loss Function ---
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_token_id)

        # --- Learning Rate Scheduler (Warmup + Decay) ---
        self.scheduler = None # Initialize as None
        self.num_training_steps = 0
        self.num_warmup_steps = 0

        # Setup scheduler only if in training mode (train_loader is provided)
        if self.train_loader is not None:
            # Calculate total training steps
            # Ensure config contains 'epochs' and 'batch_size' or handle appropriately
            if 'epochs' not in config or 'batch_size' not in config:
                 print("Warning: 'epochs' or 'batch_size' not found in config. Cannot determine total training steps for scheduler.")
            else:
                try:
                    # Estimate steps per epoch. Handle potential errors if train_loader is empty.
                    num_update_steps_per_epoch = len(self.train_loader) if len(self.train_loader) > 0 else 1
                    self.num_training_steps = num_update_steps_per_epoch * config['epochs']
                    self.num_warmup_steps = config['warmup_steps']

                    if self.num_training_steps <= 0:
                         print(f"Warning: Calculated num_training_steps is {self.num_training_steps}. Scheduler might not work correctly.")

                    # Create the scheduler
                    self.scheduler = get_linear_schedule_with_warmup(
                        self.optimizer,
                        num_warmup_steps=self.num_warmup_steps,
                        num_training_steps=self.num_training_steps
                    )
                    print(f"Scheduler created: Warmup steps={self.num_warmup_steps}, Total estimated steps={self.num_training_steps}")

                except Exception as e:
                    print(f"Error calculating training steps or creating scheduler: {e}")
                    self.scheduler = None # Fallback to no scheduler if error occurs
        else:
             print("Running in test mode or train_loader not provided. Scheduler not created.")


        # --- Training State ---
        self.start_epoch = 0
        self.best_bleu = 0.0
        self.global_step = 0 # Tracks total training steps across epochs/resumes

        # --- wandb Configuration ---
        self.use_wandb = config.get('use_wandb', False) # Default to False if not specified
        self.wandb_run = None # Store wandb run object
        if self.use_wandb:
            wandb_id_file = os.path.join(checkpoint_dir, 'wandb_id.json')
            resume_wandb_run = config.get('resume_wandb', False) and os.path.exists(wandb_id_file)

            if resume_wandb_run:
                try:
                    with open(wandb_id_file, 'r') as f:
                        wandb_data = json.load(f)
                    self.wandb_run = wandb.init(
                        project=config.get('wandb_project', 'nmt-transformer'),
                        # name=config.get('wandb_name', 'zh-en-transformer'), # Name might not be needed if resuming by ID
                        id=wandb_data['id'],
                        resume="must",
                        config=config # Update config in case parameters changed
                    )
                    print(f"Resuming wandb run with id: {wandb_data['id']}")
                except Exception as e:
                    print(f"Failed to resume wandb run: {e}. Initializing new run.")
                    resume_wandb_run = False # Fallback to new run

            if not resume_wandb_run: # If not resuming or resuming failed
                try:
                    self.wandb_run = wandb.init(
                        project=config.get('wandb_project', 'nmt-transformer'),
                        name=config.get('wandb_name', 'zh-en-transformer'),
                        config=config
                    )
                    # Save wandb run ID for potential future resuming
                    with open(wandb_id_file, 'w') as f:
                        json.dump({'id': self.wandb_run.id}, f)
                    print(f"Initialized new wandb run with id: {self.wandb_run.id}")
                except Exception as e:
                    print(f"Failed to initialize wandb: {e}. Disabling wandb.")
                    self.use_wandb = False # Disable wandb if init fails

            # Watch model if wandb run is active
            if self.use_wandb and self.wandb_run:
                wandb.watch(model, log_freq=100) # Log gradients/parameters every 100 steps

    def _train_epoch(self, epoch):
        """Trains the model for one epoch."""
        self.model.train() # Set model to training mode
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Prepare decoder input (excluding last token) and target output (excluding first token)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # --- Forward pass ---
            self.optimizer.zero_grad() # Zero gradients before forward pass
            # Model internally creates masks based on padding and causality
            outputs = self.model(src, tgt_input)

            # --- Calculate loss ---
            # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize), (Batch * SeqLen)
            loss = self.criterion(outputs.view(-1, len(self.tgt_vocab)), tgt_output.reshape(-1))

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {self.global_step}. Skipping update.")
                # Consider stopping or adding more debugging here
                continue # Skip backward/optimizer step if loss is NaN

            # --- Backward pass and Optimization ---
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])

            self.optimizer.step()

            # --- Update Learning Rate (Step-based Scheduler) ---
            if self.scheduler:
                 self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1 # Increment global step counter

            # Update progress bar postfix
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

            # Log metrics to wandb periodically
            if self.use_wandb and self.wandb_run and self.global_step % 100 == 0: # Log every 100 steps
                wandb.log({
                    'step_train_loss': loss.item(),
                    'learning_rate': current_lr,
                    # Log global_step directly if needed, wandb usually tracks its own step
                }, step=self.global_step) # Use global_step as custom step counter

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1} Training finished. Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        return avg_loss

    def evaluate(self, fast_eval=False):
        """Evaluates the model on the validation/test set."""
        self.model.eval() # Set model to evaluation mode
        total_loss = 0
        hypotheses = [] # List to store predicted token lists
        references = [] # List to store reference token lists (list of lists)

        # Determine if using fast evaluation based on method argument and config
        use_fast_eval = self.config.get('fast_eval', False) and fast_eval
        fast_eval_size = self.config.get('fast_eval_size', 1000) # Get size from config

        num_batches_to_eval = len(self.val_loader)
        eval_desc = "Evaluating (Full)"
        if use_fast_eval:
            if self.val_loader.batch_size and self.val_loader.batch_size > 0 :
                 # Calculate batches needed, ensure at least 1 batch
                 num_batches_to_eval = max(1, math.ceil(fast_eval_size / self.val_loader.batch_size))
                 eval_desc = f"Evaluating (Fast ~{fast_eval_size} samples / {num_batches_to_eval} batches)"
            else:
                 print("Warning: Cannot determine batch size for fast evaluation. Evaluating all batches.")
                 num_batches_to_eval = len(self.val_loader) # Fallback to full eval


        with torch.no_grad(): # Disable gradient calculations
            for i, batch in enumerate(tqdm(self.val_loader, desc=eval_desc, leave=False)):
                if i >= num_batches_to_eval: # Stop if limit reached (for fast_eval or full eval)
                    break

                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device) # Target needed for loss and reference

                # Calculate Loss
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                outputs = self.model(src, tgt_input) # Forward pass
                loss = self.criterion(outputs.view(-1, len(self.tgt_vocab)), tgt_output.reshape(-1))

                if not torch.isnan(loss): # Avoid adding NaN loss
                    total_loss += loss.item()
                else:
                    print(f"Warning: NaN loss detected during evaluation batch {i}. Skipping batch loss.")


                # --- Generate Translations ---
                use_beam = self.config.get('use_beam_search', False)
                max_len = self.config.get('max_len', 100)
                beam_size = self.config.get('beam_size', 5)

                if use_beam:
                    # Ensure src_padding_mask is implicitly handled or passed if needed by beam_search implementation
                    translated_ids = self.model.beam_search(src, max_len=max_len, beam_size=beam_size)
                else:
                    # Ensure src_padding_mask is implicitly handled or passed if needed by greedy_decode implementation
                    translated_ids = self.model.greedy_decode(src, max_len=max_len)

                # --- Decode and Prepare for BLEU ---
                for j in range(translated_ids.size(0)):
                    # Decode hypothesis (prediction)
                    hyp_tokens = self.tgt_vocab.decode(translated_ids[j], skip_special_tokens=True).split()
                    # Decode reference (ground truth)
                    ref_tokens = self.tgt_vocab.decode(tgt[j], skip_special_tokens=True).split()

                    hypotheses.append(hyp_tokens)
                    references.append([ref_tokens]) # NLTK expects list of references per hypothesis

        # --- Calculate BLEU Score ---
        bleu = 0.0
        if hypotheses and references:
             # Use method1 smoothing, suitable for sentence-level evaluation and shorter sentences
            smoothie = SmoothingFunction().method1
            try:
                bleu = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            except ZeroDivisionError:
                print("Warning: Division by zero encountered in BLEU calculation. Setting BLEU to 0.")
                bleu = 0.0 # Handle potential division by zero
            except Exception as e:
                 print(f"An error occurred during BLEU calculation: {e}. Setting BLEU to 0.")
                 bleu = 0.0
        else:
             print("Warning: No hypotheses or references generated for BLEU calculation.")

        avg_loss = total_loss / num_batches_to_eval if num_batches_to_eval > 0 else 0

        # --- Save Checkpoints (only in training mode, implicitly checked by `self.train_loader is not None` during init?) ---
        # We only save checkpoints if we are actually training
        if self.train_loader is not None:
             current_epoch = getattr(self, 'epoch_count', self.start_epoch) # Get current epoch if available
             is_best = bleu > self.best_bleu
             if is_best:
                 self.best_bleu = bleu
                 # Save best model checkpoint
                 save_checkpoint({
                     'epoch': current_epoch,
                     'global_step': self.global_step,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict(),
                     'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None, # Save scheduler state
                     'best_bleu': self.best_bleu,
                     # Avoid saving vocab in checkpoint if loaded separately
                     # 'src_vocab': self.src_vocab,
                     # 'tgt_vocab': self.tgt_vocab
                 }, True, self.checkpoint_dir, filename='best_model.pt')
                 print(f"New best model saved with BLEU: {self.best_bleu:.4f}")

             # Save latest checkpoint (always, for resuming)
             save_checkpoint({
                 'epoch': current_epoch,
                 'global_step': self.global_step,
                 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None, # Save scheduler state
                 'best_bleu': self.best_bleu, # Save current best BLEU in latest checkpoint too
                 # 'src_vocab': self.src_vocab,
                 # 'tgt_vocab': self.tgt_vocab
             }, False, self.checkpoint_dir, filename='checkpoint.pt')
        # --- End Checkpoint Saving ---

        return avg_loss, bleu

    def train(self, epochs):
        """Main training loop."""
        # Load checkpoint if resuming training
        if self.config.get('resume_training', False):
            self._load_checkpoint() # Load model, optimizer, scheduler, epoch, step, best_bleu

        print(f"Starting training from Epoch {self.start_epoch + 1}...")

        for epoch in range(self.start_epoch, epochs):
            self.epoch_count = epoch + 1 # Track current epoch number for saving

            # --- Train for one epoch ---
            train_loss = self._train_epoch(epoch)

            # --- Evaluate on validation set ---
            # Use fast_eval setting from config
            val_loss, bleu = self.evaluate(fast_eval=self.config.get('fast_eval', False))

            # --- Log Metrics ---
            log_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_bleu': bleu, # Use a distinct name like val_bleu
                'learning_rate': self.optimizer.param_groups[0]['lr'], # Log current LR
                'best_val_bleu': self.best_bleu, # Log historical best BLEU
                # global_step is logged periodically in _train_epoch
            }
            if self.use_wandb and self.wandb_run:
                 wandb.log(log_data) # Log per epoch results

            print(f"Epoch {epoch+1}/{epochs} Completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val BLEU: {bleu:.4f}, Best Val BLEU: {self.best_bleu:.4f}")

            # --- Optional: Early Stopping Logic ---
            # Add logic here based on validation BLEU not improving for N epochs if desired

        # --- Finish wandb run after training loop ---
        if self.use_wandb and self.wandb_run:
            print("Finishing wandb run...")
            wandb.finish()

    def _load_checkpoint(self):
        """Loads checkpoint to resume training."""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            # Pass the scheduler to load its state as well
            checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler, self.device)
            if checkpoint:
                # Load states from the returned checkpoint dictionary
                self.start_epoch = checkpoint.get('epoch', 0) # Get epoch, default to 0 if not found
                self.best_bleu = checkpoint.get('best_bleu', 0.0)
                self.global_step = checkpoint.get('global_step', 0)

                # Optimizer and scheduler states are loaded within load_checkpoint util function
                print(f"Resuming training from Epoch {self.start_epoch + 1}, Global Step: {self.global_step}, Best BLEU: {self.best_bleu:.4f}")
            else:
                print("Checkpoint loading failed (load_checkpoint returned None). Starting from scratch.")
                self.start_epoch = 0 # Ensure starting from scratch if loading fails
                self.best_bleu = 0.0
                self.global_step = 0
        else:
            print("No checkpoint found at {checkpoint_path}. Starting training from scratch.")
            self.start_epoch = 0
            self.best_bleu = 0.0
            self.global_step = 0