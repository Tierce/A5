import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)
from torch.utils.data.dataloader import DataLoader

import dataset
import model as model_file
import trainer as trainer_file
import attention as attention_file
import utils
import math
import logging

logger = logging.getLogger(__name__)

argp = argparse.ArgumentParser()
argp.add_argument('function',
    help="Whether to pretrain, finetune or evaluate a model",
    choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
    help="Which variant of the model to run ('vanilla' or 'synthesizer')",
    choices=["vanilla", "synthesizer"])
argp.add_argument('pretrain_corpus_path',
    help="Path of the corpus to pretrain on", default=None)
argp.add_argument('--reading_params_path',
    help="If specified, path of the model to load before finetuning/evaluation",
    default=None)
argp.add_argument('--writing_params_path',
    help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
    help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the 
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path,encoding='utf-8').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the synthesizer models
mconf = model_file.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256)

"""
Don't change above here; write your code below
"""

if args.variant == 'vanilla':
    pass # TODO [part c]: Make some model here
    
    class CausalSelfAttention(nn.Module):
    # """
    # A vanilla multi-head masked self-attention layer with a projection at the end.
    # It is possible to use torch.nn.MultiheadAttention here but I am including an
    # explicit implementation here to show that there is nothing too scary here.
    # """

        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            # key, query, value projections for all heads
            self.key = nn.Linear(config.n_embd, config.n_embd)
            self.query = nn.Linear(config.n_embd, config.n_embd)
            self.value = nn.Linear(config.n_embd, config.n_embd)
            # regularization
            self.attn_drop = nn.Dropout(config.attn_pdrop)
            self.resid_drop = nn.Dropout(config.resid_pdrop)
            # output projection
            self.proj = nn.Linear(config.n_embd, config.n_embd)
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))
            self.n_head = config.n_head

        def forward(self, x, layer_past=None):
            B, T, C = x.size()

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

            # output projection
            y = self.resid_drop(self.proj(y))
            return y
    
    
    class Block(nn.Module):
    # """ an unassuming Transformer block """

        def __init__(self, config):
            super().__init__()
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    class GPT(nn.Module):
        """  the full GPT language model, with a context size of block_size """

        def __init__(self, config):
            super().__init__()

            # input embedding stem
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
            self.drop = nn.Dropout(config.embd_pdrop)
            # transformer
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
            # decoder head
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            self.block_size = config.block_size
            self.apply(self._init_weights)

            logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        def get_block_size(self):
            return self.block_size

        def _init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        def configure_optimizers(self, train_config):
            """
            This long function is unfortunately doing something very simple and is being very defensive:
            We are separating out all parameters of the model into two buckets: those that will experience
            weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
            We are then returning the PyTorch optimizer object.
            """

            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # special case the position embedding parameter in the root GPT module as not decayed
            no_decay.add('pos_emb')

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
            return optimizer

        def forward(self, idx, targets=None):
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."

            # forward the GPT model
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)

            # if we are given some desired targets also calculate the loss
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            return logits, loss
            
    model = GPT(mconf)
    model.to(device)
            
elif args.variant == 'synthesizer':
    pass # TODO [part g]: Make some other model here
    

    
    class Block(nn.Module):
    # """ an unassuming Transformer block """

        def __init__(self, config):
            super().__init__()
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.attn = attention_file.SynthesizerAttention(config)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    class GPTsyn(nn.Module):
        """  the full GPT language model, with a context size of block_size """

        def __init__(self, config):
            super().__init__()

            # input embedding stem
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
            self.drop = nn.Dropout(config.embd_pdrop)
            # transformer
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
            # decoder head
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            self.block_size = config.block_size
            self.apply(self._init_weights)

            logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        def get_block_size(self):
            return self.block_size

        def _init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        def configure_optimizers(self, train_config):
            """
            This long function is unfortunately doing something very simple and is being very defensive:
            We are separating out all parameters of the model into two buckets: those that will experience
            weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
            We are then returning the PyTorch optimizer object.
            """

            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # special case the position embedding parameter in the root GPT module as not decayed
            no_decay.add('pos_emb')

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
            return optimizer

        def forward(self, idx, targets=None):
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."

            # forward the GPT model
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)

            # if we are given some desired targets also calculate the loss
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            return logits, loss
            
    
    
    model = GPTsyn(mconf)
    model.to(device)
    
    
    
    

# From here on, your code should be identical independent of which
# variant (vanilla or synthesizer) has been chosen.

if args.function == 'pretrain':
    assert args.pretrain_corpus_path is not None
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters for pretraining:
    #     max_epochs=650
    #     batch_size=128
    #     learning_rate=6e-3
    #     lr_decay=True
    #     warmup_tokens=512*20
    #     final_tokens=200*len(pretrain_dataset)*block_size
    #     num_workers=4
    # raise NotImplementedError
    from dataset import CharCorruptionDataset
    mypretraindata = CharCorruptionDataset(open(args.pretrain_corpus_path,encoding='utf-8').read(),block_size = block_size)
    tconf = trainer_file.TrainerConfig(max_epochs=650, batch_size=128, learning_rate=6e-3,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                      num_workers=0)
    trainer = trainer_file.Trainer(model, mypretraindata, None, tconf)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)
    print("pre training complete")
    
    
    
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #     Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    from dataset import NameDataset
    mynamedata = NameDataset(pretrain_dataset,open(args.finetune_corpus_path,encoding='utf-8').read())

    class TrainerConfig:
        # optimization parameters
        max_epochs = 10
        batch_size = 64
        learning_rate = 3e-4
        betas = (0.9, 0.95)
        grad_norm_clip = 1.0
        weight_decay = 0.1 # only applied on matmul weights
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        lr_decay = False
        warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
        final_tokens = 260e9 # (at what point we reach 10% of original LR)
        # checkpoint settings
        ckpt_path = None
        num_workers = 0 # for DataLoader

        def __init__(self, **kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)

    class Trainer:

        def __init__(self, model, train_dataset, test_dataset, config):
            self.model = model
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.config = config

            # take over whatever gpus are on the system
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()
                self.model = torch.nn.DataParallel(self.model).to(self.device)

        # def save_checkpoint(self):
            # DataParallel wrappers keep raw model object in .module attribute
            # raw_model = self.model.module if hasattr(self.model, "module") else self.model
            # logger.info("saving %s", self.config.ckpt_path)
            # torch.save(raw_model.state_dict(), self.config.ckpt_path)

        def train(self):
            model, config = self.model, self.config
            raw_model = model.module if hasattr(self.model, "module") else model
            optimizer = raw_model.configure_optimizers(config)

            def run_epoch(split):
                is_train = split == 'train'
                model.train(is_train)
                data = self.train_dataset if is_train else self.test_dataset
                loader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

                losses = []
                pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
                for it, (x, y) in pbar:

                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        logits, loss = model(x, y)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    if is_train:

                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

                if not is_train:
                    test_loss = float(np.mean(losses))
                    logger.info("test loss: %f", test_loss)
                    return test_loss

            best_loss = float('inf')
            self.tokens = 0 # counter used for learning rate decay
            for epoch in range(config.max_epochs):

                run_epoch('train')
                if self.test_dataset is not None:
                    test_loss = run_epoch('test')

                # supports early stopping based on the test loss, or just save always if no test set is provided
                good_model = self.test_dataset is None or test_loss < best_loss
                if self.config.ckpt_path is not None and good_model:
                    best_loss = test_loss
                    # self.save_checkpoint()
    
    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path))
        tconf = trainer_file.TrainerConfig(max_epochs=10, batch_size=256, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                          num_workers=0)
        trainer = trainer_file.Trainer(model, mynamedata, None, tconf)
        trainer.train()
        torch.save(model.state_dict(), args.writing_params_path)
        print("fine tuning complete")
    else:
        tconf = trainer_file.TrainerConfig(max_epochs=75, batch_size=256, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                          num_workers=0)
        trainer = trainer_file.Trainer(model, mynamedata, None, tconf)
        trainer.train()
        torch.save(model.state_dict(), args.writing_params_path)
        print("training complete")
    
    
    
    
    
    # raise NotImplementedError
elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w',encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path,encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))

