from LLMHelper import *
from ModelParams import *
params = ModelArgs()

#Feedforward Layer

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float],):
        super().__init__()
        # custom dim factor multiplier that ensures we're using a multiple of 256, likely for hardware efficiency reasons
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


#Repeating Block
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, device)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout_rate = args.dropout_rate

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: int = None, training = False, ):
        # our two residual connections, plus dropout which will only happen if we're training
        #RESIDUAL CONNECTION 1
        h = x # save the inputs for the residual connection
        #apply RMS Norm to the inputs
        x = self.attention_norm(x)
        #apply the attention layer
        x = self.attention(x, freqs_cis, mask, start_pos)
        #Apply the Dropout ONLY IF WE'RE IN THE TRAINING LOOP
        x = F.dropout(x, p=self.dropout_rate, training=training)
        #make the residual connection
        h = h + x

        #RESIDUAL CONNECTION 1
        out = h # save the inputs for the residual connection
        #apply RMS Norm to the inputs
        h = self.ffn_norm(h)
        #apply the feed forward layer
        h = self.feed_forward(h)
        #Apply the Dropout ONLY IF WE'RE IN THE TRAINING LOOP
        h = F.dropout(h, p=self.dropout_rate, training=training)
        #make the residual connection
        out = out + h

        return out

#the model
class LLM(nn.Module):
    def __init__(self, params: ModelArgs, tokenizer, device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_seq_len = params.max_seq_len
        self.tokenizer = tokenizer
        self.device = device

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, device))

        # final norm and linear layer
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim,
            params.vocab_size,
            bias=False)
        
        # weight sharing scheme
        # this ensures that the embedding layer and the output layers share weights
        self.tok_embeddings.weight = self.output.weight

        # precompute RoPE frequencies
        self.freqs_cis = precompute_rotary_embeddings(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            self.device,
            params.rope_theta,)

        # precompute the causal attention mask
        mask = create_causal_mask(params.max_seq_len, self.device)

        self.register_buffer('mask', mask, persistent=True) #persistent ensures this gets saved in the state_dict

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, # specifically for training. this is what you saw in section 1
                tokens: torch.Tensor,
                targets: torch.Tensor):
        bsz, seqlen = tokens.shape
        assert tokens.shape == targets.shape
        assert seqlen == self.max_seq_len

        # initialize the first residual state
        h = self.tok_embeddings(tokens)

        # grab precomputes freqs_cis
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        # run the residual state through each layer
        for layer in self.layers:
            h = layer(
                h,
                freqs_cis,
                self.mask,
                start_pos = None,
                training = True
            )

        # norm the final output then get the logits
        h = self.norm(h)
        logits = self.output(h).float()

        loss = self.criterion(
            logits.view(bsz * seqlen, self.vocab_size),
            targets.reshape(bsz * seqlen))

        return logits, loss

    @torch.inference_mode()
    def forward_inference(self,
                          tokens: torch.Tensor,
                          start_pos: int,
                          max_context_window: int,
                         ):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = self.mask[:seqlen, :seqlen]
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
        ).type_as(h)
        #print(seqlen)
        #print(start_pos)
        #print( mask.shape)


        for layer in self.layers:
            h = layer(
                h,
                freqs_cis,
                mask,
                start_pos = start_pos
            )
        h = self.norm(h)
        logits = self.output(h).float()
        return logits

    @torch.inference_mode() # no need to keep track of gradients during inference
    def Sampler(
        self,
        logits: torch.Tensor, # shape (batch_size, input_len, vocab_size)
        temperature: float, # controls how boring vs random the outputs should be
        top_p: float, # the maximum cumulative probability of output options we're willing to consider
        top_k: int, # the maximum number of output options we're willing to consider
    ) -> torch.Tensor:
        """
        The Sampler function is responsible for generating token predictions
        It supports temperature scaling, top-p (nucleus) sampling, and top-k sampling
        """
        # Select the last element for each sequence.
        logits = logits[:,-1,:] # (batch_size, input_len, vocab_size) -> (batch_size, vocab_size)

        # Apply temperature scaling
        logits.div_(temperature) # (batch_size, vocab_size) / float -> (batch_size, vocab_size)

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float) # dim=-1 is the vocab_size dimension that we calculate along

        # sort the probabilities to for use in top-p & top-k. both are (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        ### calculating top-p
        # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # mask where 0's are top-p selections & 1's are to be excluded
        top_ps_mask = (probs_sum - probs_sort) > top_p
        # the original probabilities with excluded tokens changed to 0.0
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        ### calculating top_k
        # create a shape (vocab_size) tensor that just iterates up by 1's
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        # expand our mask along the batch_size dimension to become size (batch_size, vocab_size)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        # top_ks is a list of integers. we keep whichever entries in top_ks_mask are greater than their corresponding entries in top_ks
        top_ks_mask = top_ks_mask >= top_k

        # we'll be combining top-p with top-k and using whichever gives us fewer tokens. a very conservative approach
        # this trims probs_sort to also fit within our top_k requirement
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization so that total probabilities add up to 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # now we rearrange the modified probabilities in probs_sort back to their original order according to probs_idx
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)

        return next_token_id # returns the predicted token

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_gen_len: int = None,
        temperature: float = 0.6, # default value in meta's code
        top_p: float = 0.9, # default value in meta's code
        top_k: int = params.vocab_size, # meta's code doesn't bother with topk
    ) -> str:
        """ Wrapper around sampler() that deals with manipulation of the sequence """

        max_context_window = self.max_seq_len 
 
        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)

        if max_gen_len is None:
            max_gen_len = self.max_seq_len - len(tokens)
        elif max_gen_len + len(tokens) > self.max_seq_len:
            print(f'capping max_gen_len at max_seq_len={self.max_seq_len} including input\n')
            max_gen_len = self.max_seq_len - len(tokens)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=self.device)
        tokens = tokens.unsqueeze(0) if len(tokens.shape)==1 else tokens # jic we need to add a batch dimension

        # the offset used for kv caching
        start_pos = max(tokens.shape[1] - max_context_window, 0)

        for i in range(max_gen_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits = self.forward_inference(
                tokens[:,-max_context_window:],
                start_pos = start_pos,
                max_context_window = max_context_window
            )

            # sample the next token to be used from the logit distribution
            next_token = self.Sampler(
                logits = logits,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k
            )

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

            # iterate the offset used in kv caching
            if tokens.shape[1] >= max_context_window:
                start_pos += 1

        # decode our list of tokens to an actual string
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output
