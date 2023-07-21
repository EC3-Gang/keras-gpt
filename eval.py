"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import keras_core as keras
from model import GPTConfig, GPT, get_param

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = False
seed = 1337
device = 'gpu' 
dtype = 'float16' if device != 'cpu' else 'float32'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

keras.utils.set_random_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

if device == 'gpu':
    keras.mixed_precision.set_dtype_policy(dtype)

# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = np.random.randint(len(data) - block_size, (batch_size,))
        x = keras.ops.stack([keras.ops.convert_to_tensor((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = keras.ops.stack([keras.ops.convert_to_tensor((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = keras.ops.convert_to_tensor(np.random.randint(0, 50304, (batch_size, block_size)))
    y = keras.ops.convert_to_tensor(np.random.randint(0, 50304, (batch_size, block_size)))
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 12, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
    bias = bias,
)
model = GPT(gptconf)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95))

model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy)
model(np.zeros((batch_size, block_size))) # build the model before using it

print("Total parameters:", get_param(model))

for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
    t0 = time.time()
    X, Y = get_batch('train')
    for k in range(num_steps):
        X, Y = get_batch('train')
        history = model.fit(X, Y, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        print(f"{k}/{num_steps} loss: {loss:.4f}")
    t1 = time.time()
    dt = t1-t0
    mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
    if stage == 1:
        print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")