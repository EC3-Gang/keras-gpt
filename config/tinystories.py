out_dir = 'out-tinystories'
eval_interval = 500
eval_iters = 200
log_interval = 10 

always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'TinyStories'
wandb_run_name = '8l8h768d'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 768
dropout = 0.2

learning_rate = 5e-5 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 5e-6 # learning_rate / 10 usually
beta2 = 0.9 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1000 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
