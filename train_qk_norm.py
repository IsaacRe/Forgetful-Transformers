from tqdm.auto import tqdm
from datasets import load_dataset
import numpy as  np
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import transformers_drop_in as drop_in
import tensor_util as tu
from config import CONFIG
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--effective-batch-size", default=64, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lrd-gamma", default=0.1, type=float)
parser.add_argument("--lrd-steps", nargs="*", default=[5000, 14000], type=int)
parser.add_argument("--eval-interval", default=50, type=int)
parser.add_argument("--save-interval", default=50, type=int)
parser.add_argument("--wandb-api-key", default=None, type=str)
parser.add_argument("--experiment-id", default="QKNorm")
args = parser.parse_args()

use_wandb = args.wandb_api_key is not None
if use_wandb:
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.experiment_id)

print(args)

batch_size = args.batch_size
effective_batch_size = args.effective_batch_size
val_set_size = 512
batches_per_val = args.eval_interval
lrd_gamma = args.lrd_gamma
lrd_steps = args.lrd_steps
save_interval = args.save_interval

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
dataset = load_dataset('wikitext', 'wikitext-103-v1')
split = dataset['train']

l = np.load('token_length.npy')
assert all([len(tok) > 512 for tok in tokenizer.batch_encode_plus([split[int(i)]['text'] for i in np.where(l > 512)[0][:20]])['input_ids']])


def normalize_keys(key, value):
    keyn = key / torch.sqrt((key**2).sum(dim=-1))[:,:,:,None]
    return keyn, value

drop_in.consolidate_kv = normalize_keys

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.cuda()

def validate(indices, batch_size):
    nlls = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(np.array_split(indices, len(indices) // batch_size)):
            model_input = {name: t.to(CONFIG.device) for name, t in tokenizer.batch_encode_plus(split[batch]['text'],
                                                                                        return_tensors="pt",
                                                                                        truncation=True,
                                                                                        max_length=CONFIG.context_length).items()}
            labels = model_input['input_ids'].clone()
            out = model(**model_input, labels=labels)  # forward pass handles predicted token time shift
            nlls += [out.loss]
    loss = torch.stack(nlls).mean()
    ppl = torch.exp(loss)
    print(f"Validation: Loss={loss}, PPL={ppl}")
    model.train()
    return loss.item(), ppl.item()

batches_per_update = effective_batch_size // batch_size
optim = Adam(drop_in.GLOBALS.new_params, lr=1e-3)
scheduler = MultiStepLR(optim, gamma=lrd_gamma, milestones=lrd_steps)
log_interval = 50
loss = 0
ft_indices, = np.where(np.bitwise_and(l < CONFIG.context_length, l >= CONFIG.train_length))
ft_indices = np.random.choice(ft_indices, len(ft_indices), replace=False)
val_indices, = np.where(l >= CONFIG.context_length)
val_indices = np.random.choice(val_indices, val_set_size)
full_batch_i = 0
for i, batch in enumerate(np.array_split(ft_indices, len(ft_indices) // batch_size)):
    model_input = {name: t.to(CONFIG.device) for name, t in tokenizer.batch_encode_plus(split[batch]['text'],
                                                                                 return_tensors="pt",
                                                                                 truncation=True,
                                                                                 max_length=CONFIG.train_length).items()}
    labels = model_input['input_ids'].clone()
    out = model(**model_input, labels=labels)  # forward pass handles predicted token time shift
    (out.loss / batches_per_update).backward()
    loss += out.loss.cpu() / batches_per_update

    if (i + 1) % batches_per_update == 0:
        optim.step()
        optim.zero_grad()
        scheduler.step()
        ppl = torch.exp(loss)
        print(f"{full_batch_i}: Loss={loss}, PPL={ppl}")
        full_batch_i += 1

        wandb.log(
            {"loss": loss, "ppl": ppl},
            step=full_batch_i,
        )
        loss = 0

        if full_batch_i % batches_per_val == 0:
            val_loss, val_ppl = validate(val_indices, batch_size)
            wandb.log(
                {"val_loss": val_loss, "val_ppl": val_ppl},
                step=full_batch_i,
            )

        if full_batch_i % save_interval == 0:
            print("saving model...")
            torch.save(model.state_dict(), 'gpt2_qknorm.pt')
