"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from memory import MemoryAwareConfig, MemoryAwareInference, SQLiteMemoryStore, build_embedder
from memory.explain import format_trace
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
memory_enabled = False
memory_db_path = 'memory.sqlite'
memory_user_id = 'default'
memory_embedder = 'hash-384'
memory_top_k = 12
memory_max_items = 4
memory_similarity_threshold = 0.18
memory_critic_threshold = 0.58
memory_maybe_threshold = 0.48
memory_max_age_days = -1
memory_token_budget = 192
memory_type_allowlist = ''
memory_recent_context = ''
memory_system_prompt = ''
memory_capture_input = False
memory_capture_type = 'auto'
memory_capture_importance = 0.5
memory_version_group_id = ''
memory_explain = False
memory_prompt_style = 'auto'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    allowed_chars = set(stoi.keys())
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    allowed_chars = None
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
original_start = start

memory_system = None
memory_store = None
if memory_enabled:
    memory_store = SQLiteMemoryStore(memory_db_path)
    embedder = build_embedder(memory_embedder)
    memory_system = MemoryAwareInference(
        store=memory_store,
        embedder=embedder,
        config=MemoryAwareConfig(
            user_id=memory_user_id,
            top_k=memory_top_k,
            max_items=memory_max_items,
            similarity_threshold=memory_similarity_threshold,
            critic_threshold=memory_critic_threshold,
            maybe_threshold=memory_maybe_threshold,
            max_age_days=None if memory_max_age_days < 0 else memory_max_age_days,
            memory_token_budget=memory_token_budget,
            type_allowlist=memory_type_allowlist,
        ),
    )
    if memory_prompt_style == 'auto':
        resolved_memory_prompt_style = 'completion' if init_from.startswith('gpt2') and not load_meta else 'chat'
    else:
        resolved_memory_prompt_style = memory_prompt_style
    start, trace, _ = memory_system.prepare_prompt(
        query_text=original_start,
        recent_context=memory_recent_context,
        system_prompt=memory_system_prompt or None,
        encode=encode,
        plain_text_prompt=load_meta,
        allowed_chars=allowed_chars,
        prompt_style=resolved_memory_prompt_style,
    )
    if memory_explain:
        print(format_trace(trace))

start_ids = encode(start)
if len(start_ids) > model.config.block_size:
    print(
        f"WARNING: prompt length {len(start_ids)} exceeds block size {model.config.block_size}; "
        "generation will condition only on the final block."
    )
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

if memory_enabled and memory_capture_input:
    memory_system.remember(
        text=original_start,
        memory_type=memory_capture_type,
        importance=memory_capture_importance,
        version_group_id=memory_version_group_id or None,
    )
if memory_store is not None:
    memory_store.close()
