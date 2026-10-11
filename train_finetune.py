import os
import re
import time
import math
import random
import string
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from GPT2_Pretrained import GPT
    
torch.manual_seed(1337)
random.seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

enc = tiktoken.get_encoding('gpt2')
EOS_ID = enc.eot_token  # <|endoftext|>, id 50256 -- doubles as our "stop" signal and pad token

# -----------------------------------------------------------------------------
# Knobs
# -----------------------------------------------------------------------------
NUM_TRAIN_EXAMPLES = None   # None = full SQuAD train set (~87.6k, ~60-90 min). Set 2000 for a smoke test.
EPOCHS             = 1      # one pass is enough to learn a format; more risks memorising SQuAD style
MICRO_BATCH        = 4      # sequences per forward/backward = number of examples per micro-batch -- sized so the (B,T,50257) logits fit a T4
GRAD_ACCUM         = 4      # number of micro-batches whose gradients get accumulated before one optimiser step (number of micro-batch in one big batch used in the step - 4 * 4 = 16 sequences per batch)
MAX_SEQ_LEN        = 1024   # GPT-2's hard context limit (learned position embeddings, can't extend) 
                            # the max amount of context GPT-2 can take is 1024 tokens so this is the max context length we will use when fine-tuning
LR_MAX             = 3e-5   # ~20x below pretraining LR -- nudge, don't overwrite
LR_MIN             = 3e-6
WARMUP_STEPS       = 200
WEIGHT_DECAY       = 0.01
EVAL_EVERY         = 500    # steps between validation-loss checks + a sample generation
EVAL_BATCHES       = 20     # batches of val data per check
CHECKPOINT_EVERY   = 1000   # steps between Drive saves (~15 min apart)
N_FINAL_EVAL       = 200    # validation examples for the final exact-match / F1 score

# -----------------------------------------------------------------------------
# Checkpoint destination -- Drive on Colab, local folder otherwise
# -----------------------------------------------------------------------------
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    ckpt_dir = '/content/drive/MyDrive/gpt2_squad_ft'
except Exception:
    ckpt_dir = 'gpt2_squad_ft'   # not on Colab -- save next to the notebook
os.makedirs(ckpt_dir, exist_ok=True)
print(f"checkpoints -> {ckpt_dir}")

# -----------------------------------------------------------------------------
# Model -- start from the real pretrained GPT-2 124M via YOUR from_pretrained.
# Every weight is trainable (full fine-tuning; at 124M there is no need for
# LoRA-style tricks -- the whole model + AdamW state fits a T4 with room to spare).
# -----------------------------------------------------------------------------
model = GPT.from_pretrained('gpt2')
model.to(device)


# -----------------------------------------------------------------------------
# data: SQuAD v1.1 -> (tokens, prompt_len) pairs
# Each example becomes ONE training sequence:
#   "Context: {passage}\nQuestion: {question}\nAnswer: {answer}<|endoftext|>"
# We tokenize prompt and answer SEPARATELY so we know exactly where the answer
# begins -- that boundary is what the loss mask is built from.
# -----------------------------------------------------------------------------
def encode_example(ex):
    # ex is a row of the dataset and acts like a dictionary where it stores the context, question and answer 
    prompt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer:" # Creates the prompt for the Context - Question - Answer that we want GPT-2 to return
    # leading space on the answer matters: GPT-2's BPE merges " Paris" as one
    # token but "Paris" after ":" would tokenize differently -- and at inference
    # the model naturally generates the space-prefixed form after "Answer:"
    answer = ' ' + ex['answers']['text'][0] # SQuAD provides possibly multiple acceptable answers (for eval) but for training just pick the first at index 0
    # encode the prompt and answers
    prompt_ids = enc.encode(prompt)
    answer_ids = enc.encode(answer)
    # gets the whole data
    full = prompt_ids + answer_ids + [EOS_ID]   # EOS teaches the model to STOP after answering - returns the concatenation of the prompt, anseer and end of sentence tokens
    if len(full) > MAX_SEQ_LEN:
        return None # drop the rare over-length example
    return np.array(full, dtype=np.uint16), len(prompt_ids) # each encoded example is a tuple of a an array of the tokens and the length of the tokens for the prompt context

# Downloads and loads the squad dataset
print("loading SQuAD v1.1...")
squad = load_dataset('rajpurkar/squad')  # bare 'squad' alias no longer resolves on newer datasets versions

# accesses the training rows from squad - squad is a datasetdict which behaves like a dictionary
train_rows = squad['train'] if NUM_TRAIN_EXAMPLES is None else squad['train'].select(range(NUM_TRAIN_EXAMPLES))
train_data, dropped = [], 0
t0 = time.time()
# simple loop through every row, encode it and keep ones under the length limit (encode_example checks if its within the max sequence length limit)
for ex in train_rows:
    enc_ex = encode_example(ex)
    if enc_ex is None:
        dropped += 1
    else:
        train_data.append(enc_ex) # append to list of all the train_data which is a list of tuples containing an array of the tokens and the length of the prompt
val_data = [e for e in (encode_example(ex) for ex in squad['validation'].select(range(2000))) if e is not None] # same process but for val_data and condensed into a list comprehension with a max of 2000 examples (hardcoded)
lens = [len(f) for f, _ in train_data]
print(f"tokenized {len(train_data):,} train examples in {time.time()-t0:.0f}s "
      f"(dropped {dropped} over {MAX_SEQ_LEN} tokens) | seq len avg {np.mean(lens):.0f}, max {max(lens)}")

# LENGTH-SORTED BUCKETING: group similar-length examples into the same
# micro-batch so padding (and the padded logits tensor) stays small. We sort
# once, slice into micro-batches, then shuffle the BATCH ORDER each epoch --
# randomness across steps is preserved, only batch composition is non-random.
train_data.sort(key=lambda e: len(e[0])) # e[0] = tokens array so sorting by length of the token array
val_data.sort(key=lambda e: len(e[0]))
micro_batches = [train_data[i : i + MICRO_BATCH]
                 for i in range(0, len(train_data) - MICRO_BATCH + 1, MICRO_BATCH)] # list of lists for micro batch - each of these lists stores the data as (token_array, len(prompt))


# -----------------------------------------------------------------------------
# Batching + the Loss Mask (the conceptual heart of this cell)
# -----------------------------------------------------------------------------
def make_batch(examples):
    maxlen = max(len(full) for full, _ in examples) - 1 # -1 because of the x/y shift where a token predicts the next - maxlen = longest context sequence in the batch
    x = torch.full((len(examples), maxlen), EOS_ID, dtype=torch.long) # pre-allocates the input tensor to be filled entirely with EOS tokens
    y = torch.full((len(examples), maxlen), -100, dtype=torch.long) # pre-allocates the label tensor to be filled entirely with -100 as this means ignore this position in the loss 
                                                                    # (-100 is used for the prompt as we don't use it to evaluate the loss, only the answer)
    # i = index, (full, plen) = tuple of array and prompt length
    for i, (full, plen) in enumerate(examples):
        seq = torch.from_numpy(full.astype(np.int64)) # tales the token list (full) and converts each element into an in64 from uint16
        n = len(seq) - 1 # how many (input, target) pairs this particular token sequence contributes - formula is len - 1
        x[i, :n] = seq[:-1] # indexes into the ith row of the tensor x and assigns the input row as every token except the last (up to n as length of n will vary depending on the token sequence)
                            # x by default has a maxlen number of columns to deal with the longest possible token sequence, but for token sequences that are shorter, only change values up to its length - the rest after remain as EOS tokens
        yi = seq[1:].clone() # for the ith row the outputs are just x shifted by 1 since we are using next-character prediction inputs and labels
        yi[: plen - 1] = -100 # mask the prompt - from the start to the end of the prompt - we mask the token as -100 
                              # this means we do not include tokens that are already part of the prompt in the loss calculations as we only care about the answer
                              # the prompt is something already given to the model and therefore it does not "generate" anything when it is returned to the user - what's relevant is the answer
        y[i, :n] = yi # assign this to the actual tensor y
    return x, y
# F.cross_entropy averages over the non-masked positions only, so the printed 
# loss is "average loss per ANSWER token" -- directly comparable across batches.
# -100 is a sentinel value chosen by PyTorch that when encountered indicates to skip it when calculating CEL


# -----------------------------------------------------------------------------
# Inference Helper
# ask() is essentially the generate() function used in GPT-2 but with greedy
# decoding instead of top-k sampling and with explicit EOS_ID to stop generation
# This exact function is what your RAG pipeline will call.
# Greedy decoding (argmax), not sampling: for extractive QA we want the single
# most likely answer, not creative variety. Stops at EOS = the stop behaviour
# we are about to train in.
# -----------------------------------------------------------------------------
@torch.no_grad()
def ask(context, question, max_new_tokens=48):
    model.eval()
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"   # MUST match the training template
    ids = enc.encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0) # unsqueeze(0) adds another dim of value 1 at position 0 (the front) because forward() in GPT requires a batch dim
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -MAX_SEQ_LEN:]) # defensive cropping in case a very long context pushes the prompt near the 1024 limit (same trick as your original generate).
        next_id = int(logits[0, -1].argmax()) # greedy decoding which calculates the next_id as the most likely token - deterministic rather than probabilistic 
                                              # for QA we want the most likely answer, not creative variety 
        if next_id == EOS_ID: # whole point is to train the model to know when to stop generation cleanly
            break
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1) # keep appending the predicted tokens 
                                                                                # a 2D tensor of (1, sequence_length) has tokens appended to the end of it [[1, 2, 3]] + [[4]] (appended along dim 1 - to the side - columns)
    return enc.decode(idx[0, len(ids):].tolist()).strip() #  slices off just the newly generated tokens (everything after the original prompt length) and decodes back to text

# a fixed demo question, asked before/during/after training to SEE the behaviour change
DEMO = squad['validation'][0]


# -----------------------------------------------------------------------------
# optimizer + schedule + fp16 machinery
# fp16 tensor cores are the T4's fast path, and GradScaler protects tiny fp16
# gradients from underflowing to zero
# -----------------------------------------------------------------------------
steps_per_epoch = len(micro_batches) // GRAD_ACCUM 
# 1 epoch consists of training over all big batches and 1 big batch is split into multiple micro-batches
# 1 step occurs for every big batch = number of big batches = no micro-batches in total // no micro-batch per big batch
total_steps = EPOCHS * steps_per_epoch # 1 epoch trains over all big batches so we can think total steps as repeating this training over all big batches
print(f"plan: {total_steps:,} optimizer steps ({EPOCHS} epoch(s)), "
      f"effective batch {MICRO_BATCH * GRAD_ACCUM} ({GRAD_ACCUM} micro-batches of {MICRO_BATCH})")

# plain AdamW over all params -- at fine-tuning LRs the decay/no-decay split
# from pretraining barely matters, so we keep it simple
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler(device, enabled=(device == 'cuda')) # fp16 gradients can underflow to zero for every small values so GradScale multiplies loss by a large factor before backprop so gradients don't vanish


# Cosine Decay applied to the learning rate 
def get_lr(it):
    if it < WARMUP_STEPS:
        return LR_MAX * (it + 1) / WARMUP_STEPS
    ratio = (it - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS) # ratio of progress to reaching total_steps from 0 to 1
                                                                     # (iteration - warmup) / (total - warmup) = how much has it progressed out of total progression (max is a safety net in case division by 0)
    return LR_MIN + 0.5 * (1 + math.cos(math.pi * ratio)) * (LR_MAX - LR_MIN) # learning rate decreases as ratio approaches 1 since the cosine portion will go from 1 to -1 meaning lr decreases

@torch.no_grad()
def eval_val_loss():
    model.eval()
    total = 0.0 # accumulate the sum of losses
    # iterate over each of the batches in the evaluation data 
    # evaluation data is chunked into batches, the loss is calculated for each, accumulated and then average loss is computed at the end
    for k in range(EVAL_BATCHES):
        batch = val_data[k * MICRO_BATCH : (k + 1) * MICRO_BATCH] # slice the micro batch from the list
        # val_data is a list of tuples where each tuple is a (token_array, len(prompt))
        x, y = make_batch(batch) # we pass this micro-batch into make_batch which takes the batch and forms input, label pairs 
                                 # the length of each sequence in the micro-batch is different so they are padded to the same length 
                                 # the padded length is equal to max length sequence -1 since we require a +1 shfit between labels and pairs meaning we can only have n-1 pairs at a maximum 
                                 # masking then occurs so we only take into consideration the answer tokens, replacing all other tokens with - 100
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
            _, loss = model(x, y) # forward pass of the model with input and label x, y returns logits and loss
        total += loss.item() # accumulate loss over all bmicro-batches
    return total / EVAL_BATCHES # calculate average loss over all micro-batches
    # no backward passes or optimiser step - purely for evaluation and not raining


# saves the learned weights, the GPTConfig used to build the architecture and the step number to resume training mid-schedule
def save_checkpoint(name, step):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),   # Adam's running moments -- needed for a seamless resume
        'scaler': scaler.state_dict(),
        'config': model.config,
        'step': step,
    }, os.path.join(ckpt_dir, name))
    print(f"  [step {step}] checkpoint saved -> {name}")


# -----------------------------------------------------------------------------
# RESUME: if a checkpoint from a previous (possibly interrupted) run exists,
# load model + optimizer + scaler state and continue from that step instead
# of silently retraining from scratch. Checked in priority order: the
# always-fresh 'latest_ft.pt', then the final 'gpt2_124M_squad_ft.pt'.
# -----------------------------------------------------------------------------
start_step = 0
for ckpt_name in ('latest_ft.pt', 'gpt2_124M_squad_ft.pt'):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print(f"found existing checkpoint: {ckpt_path} -- resuming")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step']
        print(f"resumed at step {start_step}/{total_steps}")
        break
else:
    print("no existing checkpoint found -- starting fine-tuning from pretrained GPT-2")

# prints some sample predictions from the data
print("\n--- BASELINE / CURRENT STATE ---")
print(f"Q: {DEMO['question']}")
print(f"A: {ask(DEMO['context'], DEMO['question'])!r}")
print(f"(gold answer: {DEMO['answers']['text'][0]!r})")
if start_step == 0:
    print("expect rambling -- the base model has never seen this template\n")
else:
    print(f"(model already fine-tuned to step {start_step})\n")



# -----------------------------------------------------------------------------
# Training Loop -- STEP-based rather than epoch-based so resuming is simple:
# we just run (total_steps - start_step) more optimizer steps. With EPOCHS=1
# this is equivalent to applying a triple nested epoch loop but instead the 
# logic is dealt with using StopIteration in next_micro_batch
# Remember - 1 epoch iterates over every batch so once it reaches the end of
# all the batches traditionally we would advance to the next iteration of epochs
# but by flattening this into a single epoch run, we just start again and iterate
# over a shuffled array of micro_batches 
# -----------------------------------------------------------------------------
if start_step >= total_steps:
    print(f"checkpoint already at/past the planned {total_steps} steps -- nothing left to train.")
else:
    random.shuffle(micro_batches) # shuffles micro_batches - a full list of pre-chunked data groups
    mb_iter = iter(micro_batches) # wraps the micro_batches in an iter object() that remembers its position and yields one element at a time via next()

    def next_micro_batch():
        nonlocal_iter = mb_iter # local alias pointing to mb_iter
        try:
            return next(nonlocal_iter) # try iterating to the next batch
        except StopIteration: # but if we cannot then shuffle batches and creates a fresh iterator to continue iterating until we complete all total steps
            random.shuffle(micro_batches) # micro batches remain the same - shuffling micro batches means we form different groups of micro-batches (big batches)
                                          # the same raw data just arranged differently - prevents the data from overfitting to a specific batch and helps it learn the actual pattern in the overall data
            return next(iter(micro_batches))

    step = start_step # step starts wherever training left off (0 for fresh runs and the checkpoint if continuing)
    t_start = time.time()
    while step < total_steps: # begin the training loop - remember one step occurs for every big batch and each epoch iterates over all big batches
                              # => total_steps = no_big_batches x no_epochs 
        model.train() # set to training mode
        lr = get_lr(step) # get the learning rate depending on what step it is currently on
        for pg in optimizer.param_groups:
            pg['lr'] = lr # set the lr to the required lr

        # one optimizer step = GRAD_ACCUM small forward/backwards. Each loss is
        # divided by GRAD_ACCUM so the summed gradients equal the AVERAGE over
        # the effective batch -- same math as one big batch, a fraction of the memory.
        optimizer.zero_grad(set_to_none=True) # refresh gradient for each big batch
        loss_accum = 0.0
        for _ in range(GRAD_ACCUM): # iterate over every micro-batch in the big batch - there are GRAD_ACCUM micro-batches in the batch
            x, y = make_batch(next_micro_batch()) # pulls the next micro-batch and makes a batch off it - padding + (input, label) pairs + +1 shifting + masking 
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                _, loss = model(x, y)
            loss = loss / GRAD_ACCUM # get the weighted value for loss that contributes to the average loss of the micro-batch
            loss_accum += loss.item()
            scaler.scale(loss).backward() # grads ACCUMULATE across micro-batches - scaling applied to prevent vanishing gradients
        scaler.unscale_(optimizer) # un-scale so clipping sees true magnitudes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # computes the total norm and rescales down if that norm (magnitude) exceeds 1.0 to protect against any single unusually bad batch
        scaler.step(optimizer) # weight update (AdamW) but GradScaler wraps it with a safety check - if any gradient turned out to be inf or nan then skip it 
        scaler.update() # adjusts GradSCaler's internal scale factor for the next iteration
        step += 1 # next step = next big batch


        #-------------------------------------------------
        # Printing Diagnostics and Saving Training Results
        #-------------------------------------------------
        # print diagnostics such as eta, evaluation loss and sample
        if step % 50 == 0:
            elapsed = time.time() - t_start
            done = step - start_step
            eta_min = elapsed / max(1, done) * (total_steps - step) / 60
            print(f"step {step:5d}/{total_steps} | loss {loss_accum:.4f} | lr {lr:.2e} | ETA {eta_min:.0f} min")
        if step % EVAL_EVERY == 0:
            vl = eval_val_loss()
            print(f"  [step {step}] val loss {vl:.4f}")
            print(f"  [step {step}] demo: {ask(DEMO['context'], DEMO['question'])!r}")
        # save the weights to a pt file just in case training fails midway
        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint('latest_ft.pt', step)

    # once the entire training loop is complete save the weights to a new pt file and print the training time
    save_checkpoint('gpt2_124M_squad_ft.pt', step)
    print(f"\ntraining done in {(time.time()-t_start)/60:.0f} min")


# -----------------------------------------------------------------------------
# Final Evaluation: exact match + F1, the standard SQuAD metrics.
# Both compare NORMALIZED text (lowercase, no punctuation/articles) against
# every acceptable gold answer and take the best score:
#   EM = 1 if the prediction matches a gold answer exactly, else 0
#   F1 = word-overlap harmonic mean -- partial credit ("Queen Victoria" vs "Victoria")
# The absolute numbers will sit far below modern models -- what matters for the
# portfolio is the before/after: the baseline scores ~0 because it rambles.
# -----------------------------------------------------------------------------
# clean the text
def normalize(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())
# calculate the f1 score
def f1_score(pred, gold): # predicted answer and correct answer passed as args
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values()) # counts shared words between prediction and gold, capping repeats at whichever has fewer
                                                       # Counter() creates a dict of counts for each word and & applies an intersection for matching words
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g) # precision = TP / (TP + FP), recall = TP / (TP + FN)
                                                           # TP = overlap since it's what is correct and actually found
    return 2 * precision * recall / (precision + recall) # calculate the f1 score (harmonic mean)

# Evaluate our model
print(f"\nevaluating on {N_FINAL_EVAL} held-out examples...")
em_total, f1_total = 0.0, 0.0
eval_rows = squad['validation'].select(range(2000, 2000 + N_FINAL_EVAL))
for ex in eval_rows:
    # key structural feature of the data is they are dict-like objects meaning we can index into them to get corresponding values for a given key
    pred = ask(ex['context'], ex['question'])
    golds = ex['answers']['text']
    em_total += max(float(normalize(pred) == normalize(g)) for g in golds)
    f1_total += max(f1_score(pred, g) for g in golds)
print(f"exact match: {100*em_total/N_FINAL_EVAL:.1f}% | F1: {100*f1_total/N_FINAL_EVAL:.1f}%")