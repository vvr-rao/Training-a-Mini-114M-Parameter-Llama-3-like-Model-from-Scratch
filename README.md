# Training-a-Mini-114M-Parameter-Llama-3-like-Model-from-Scratch

Code to train a 114 Million Parameter LLM with an architecture similar to LLama.

Explanation of files
LLMHelper.py - reusable functions for Rotary Embeddings, RMS NOrm, the Attention Layer, Causal Mask.
S3Helper.py - reusable functions to upload and download files to AWS S3
ModelParams.py - Model Parameters
Model.py - the Actual model
train.py - the training code.

To train - use python train.py OR torchrun --standalone --nproc_per_node=<NUM_GPUs> train.py

For an explanation and addtional information, take a look at my blog here: https://medium.com/@venkat.ramrao/training-a-mini-114m-parameter-llama-3-like-model-from-scratch-97525185aa9c

