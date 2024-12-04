---
layout: post
title: Notes for NLP interview
date: 2024-11-04 19:52 +0800
categories: [Study, NLP]
comment: false
math: true
tags: [notes,ai,nlp,interview]
---
##  Transformers
### Transformers 结构
![transformer-architecture](/images/notes/transformer-architecture.png){:w="400" h="700"}
![transformer-attention-formulation](/images/notes/transformer-attention-formulation.png)


### Multi-head Attention 源码解析
```python
# from transformers.models.modeling_bart
class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
```
init 函数

```python
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous() 
        # ps: tensor的维度是(bsz,seq_len,hidden_sz)，hidden_sz = num_heads*head_dim, 使用view将tensor转为(bsz,seq_len,num_heads,head_dim)，transpose调换第一维和第二维，变为(bsz, num_heads, seq_len, head_dim)，便于后续处理不同头的注意力
```
+ tensor的维度是(bsz,seq_len,hidden_sz)，hidden_sz = num_heads*head_dim, 使用view将tensor转为(bsz,seq_len,num_heads,head_dim)，transpose调换第一维和第二维，变为(bsz, num_heads, seq_len, head_dim)，便于后续处理不同头的注意力

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        # encoder_outputs 就是encoder_hidden_states (batch_sz,encoder_seq_len,hidden_sz)
        past_key_value: Optional[Tuple[torch.Tensor]] = None, 
        # decoder使用cross-attention，需要encoder_outputs作为key和value。decoder在生成时需要重复调用forward，而每次forward中key和value都是一样的，所以只需要在第一次forward时将encoder_outputs根据k_proj和v_proj以及多头分割转变为key和value，然后将其保存在past_key_value中，就可重复使用，避免了每次forward都重复调用k_proj和v_proj进行计算。在生成时有效提升推理性能。
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None 
        # ps: Encoder is self-attention, decoder requires cross-attention

        bsz, tgt_len, _ = hidden_states.size() # ps: (batch_sz,decoder_seq_len,hidden_size)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling  # ps: Q = W_q * hidden_states:  (batch_sz,decoder_seq_len,hidden_size) 
```
+ key_value_states就是encoder_hidden_states (batch_sz,seq_len,hidden_sz)
+ decoder使用cross-attention，需要encoder_outputs作为key和value。decoder在生成时需要重复调用forward，而每次forward中key和value都是一样的，所以只需要在第一次forward时将encoder_outputs根据k_proj和v_proj以及多头分割转变为key和value，然后将其保存在past_key_value中，就可重复使用，避免了每次forward都重复调用k_proj和v_proj进行计算。在生成时有效提升推理性能。

```python
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # ps: past_key_value是encoder_hidden_states经过k_proj和v_proj以及_shape分割之后的结果(batch_sz,num_heads,encoder_seq_len,head_dim)， key_value_states就是encoder_hidden_states,为(batch_sz,encoder_seq_len,hidden_sz)
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            # ps: cross attention需要计算key和value
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        # 前面是decoder中的cross-attention部分，下面是decoder中的self-attention部分
        elif past_key_value is not None:
            # ps: 在self-attention的情况下，如果past_key_value不为空，说明至少是第二次forward。masked self attention每一步forward需要使用的前面的key和value，推导见后，因此可以通过保存每步的key和value节省计算开销(kv cache)
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
```
+ 在self-attention的情况下，如果past_key_value不为空，说明至少是第二次forward。masked self attention每一步forward需要使用的前面的key和value，推导见后，因此可以通过保存每步的key和value节省计算开销(kv cache) ，适用于decoder的masked self-attention
+ K-V derivation $Q,K,V$:(batch_sz* num_heads,seq_len, head_dim), $K_i,V_i$:(batch_sz* num_heads,1, head_dim)对应第i个token(start_token为第1个token)的embedding(batch_sz,1,hidden_sz)经过如上操作后得到的KV， 生成第i+1个token时，seq_len已经为i，前i个token的embedding组成了seq embedding为(batch_sz,i,hidden_sz)，经过上述操作后得到当前的$K,V$Y由$K_j,V_j,j=1,...,i$在seq_len维度上拼接而成。

$$
Attn_1(Q,K,V)  = softmaxed(Q_1K_1^T)V_1 
$$
softmax的矩阵为(-1,1,1)
$$
\begin{align*}
Attn_2(Q,K,V) & = softmaxed( Q_2(K_1^T, K_2^T)) (V_1,V_2)^T \\
& = softmaxed(Q_2 K_1^T ,Q_2 K_2^T)(V_1,V_2)^T
& = softmaxed(Q_2 K_1^T ,Q_2 K_2^T)
\end{align*}
$$
注意只拆分$K,V$为$k_i,V_i$,$Q$不拆，softmax的矩阵为(-1,2,2),softmax作用于key的seq_len维度，即上式中的行。(为什么cat hidden_state?)

```python
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        # ps: (batch_sz, num_heads, seq_len, head_dim) -> (batch_sz*num_heads, seq_len, head_dim) 合并前两个维度便于后续计算，因为合并之后减少索引次数带来的额外开销，同时合并成一个大批次的数据后，内存连续，也方便传输给计算设备进行计算

        src_len = key_states.size(1) # ps: seq_len
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # torch.bmm为批量矩阵乘法 Q:(batch_sz*num_heads,decoder_seq_len,head_dim) * K^T:(batch_sz*num_heads,head_dim,encoder_seq_len) -> attn_weights:(batch_sz*num_heads,seq_len,seq_len)
```
+ (batch_sz, num_heads, seq_len, head_dim) -> (batch_sz*num_heads, seq_len, head_dim) 合并前两个维度便于后续计算，因为合并之后减少索引次数带来的额外开销，同时合并成一个大批次的数据后，内存连续，也方便传输给计算设备进行计算
+ torch.bmm为批量矩阵乘法 Q:(batch_sz * num_heads,decoder_seq_len,head_dim) * K^T:(batch_sz* num_heads,head_dim,encoder_seq_len) -> attn_weights:(batch_sz*num_heads,decoder_seq_len,encoder_seq_len)

```python
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        # 注意这里tgt_len即decoder_seq_len和src_len即encoder_seq_len在生成时往往是不同的，tgt_len每次forward都会增加，src_len仅为input的seq_len，是保持不变的
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
```
+ 注意这里tgt_len即decoder_seq_len和src_len即encoder_seq_len在生成时往往是不同的，tgt_len每次forward都会增加，src_len仅为input的seq_len，是保持不变的
+ attention_weights加上attention_mask后经过一层softmax再乘value，attention_mask包括padding_mask和causal_mask,padding mask用于batch内的对齐，causal_mask在decoder的self-attention训练时使用，维度为(bsz, 1, tgt_len, tgt_len)，为下三角矩阵，causal[i][j]表示token i对token j的注意mask，当i>=j时为1，表明token i可以注意到之前的token j，否则为0。
+ 在实际计算时，attention_mask中的1位置被替换成0，0位置被替换成-inf，这样在softmax之后该位置的分数基本为0

```python
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # 利用broadcast机制进行矩阵点积，layer_head_mask中可能需要mask掉某些注意力头（即对应值为0），broadcast后与attn_weights相乘可以直接让该head维度的所有值都为0
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # dropout 之后与value相乘

        attn_output = torch.bmm(attn_probs, value_states)
        # (bsz*num_heads,decoder_seq_len,encoder_seq_len) * (bsz*num_heads,encoder_seq_len,head_dim) -> (bsz*num_heads,decoder_seq_len,head_dim)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        # 复原至(bsz,decoder_seq_len,num_heads,head_dim)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        # 复原至(bsz,decoder_seq_len, hidden_sz = num_heads * head_dim)

        attn_output = self.out_proj(attn_output)
        # 输出再proj一下，shape不变

        return attn_output, attn_weights_reshaped, past_key_value
```
+ 利用broadcast机制进行矩阵点积，layer_head_mask中可能需要mask掉某些注意力头（即对应值为0），broadcast后与attn_weights相乘可以直接让该head维度的所有值都为0
+ 最后输出仍然要proj一下，shape不变

### Encoder Layer源码解析
```python
class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function] # 激活函数
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
```
init 函数
```python
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
```
### Decoder Layer 源码解析
```python
class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
```
init 函数
+ 注意GPT模型虽然称作decoder-only，但是实际上只保留了multi-head masked self-attention，删去了cross-attention部分

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
```
self-attention part
```python
        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
```

### GenerationMixin 源码解析
transformers中generationmixin是pretrainedmodel的父类之一，所有生成的方法在此定义。

#### **generate方法定义**
```python
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
```
generate方法, 参数详解:
+ inputs作为生成的prompt或encoder的输入
+ generation_config生成的参数，如do_sample, num_beams, temperature
+ logits_processor 自定义logits处理模块
+ stopping_criteria 自定义生成停止策略
+ prefix_allowed_tokens_fn 限制每步的beam search仅关注在allowed tokens当中，见[Autoregressive EntityRetrieval](https://arxiv.org/abs/2010.00904)
+ synced_gpus 是否继续while loop，Unless overridden, this flag will be set to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`
+ assistant_model 一个用于辅助生成的小模型，通过替代大模型预测下一个token来加速推理
+ streamer 用于处理输出sequences

#### **generate-预检查**
```python
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class() # 检查模型是否支持generate
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation
        # ps: kwargs.pop(k,default) -> v，如果找到(k,v)，返回v，否则返回default

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)
        # 做一些预检查

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys()) # 通过签名检查forward是否支持attention_mask
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
```
这部分是一些预先准备工作
```python
        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        ) # 准备模型输入
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )   
```
这部分是对模型输入的准备，调用了_prepare_model_inputs用于获取模型输入，如果指定了inputs，这里没有变化，如果没有指定inputs，这里返回的shape为(batch_sz,encoder_seq_len/1) 分别对应encoder-decoder和decoder-only

#### **_prepare_model_inputs方法**
```python
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}
        # 提取所有可能和生成相关的的参数

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg
        # 有可能从kwargs中输入了

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs
```
其中调用了_maybe_initialize_input_ids_for_generation，如下

#### **_maybe_initialize_input_ids_for_generation方法**
```python
    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs") # shape为(batch_sz,encoder_seq_len,hidden_sz)
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.last_hidden_state.size()[:-1] # 即(batch_sz, encoder_seq_len)
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100
        # 对于encoder-decoder模型的decoder部分，第一次输入forward时用encoder_outputs输入给decoder，但是需要将它们的input_ids都设为IGNORE_INDEX=-100

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id
```
+ 如果已经有inputs了，直接返回，下面的处理仅针对没有inputs的情况
+ 注意这里对于encoder-decoder模型的decoder部分，第一次输入forward时用encoder_outputs输入给decoder，但是需要将它们的input_ids都设为IGNORE_INDEX=-100
+ 而对于decoder-only模型,返回的是batch_sz个bos_token_id，作为输出的start token，如果已经指定了inputs_embeds，则啥都不返回

#### **generate-模型参数补全**
```python
        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )
```
+ 这里调用_prepare_attention_mask_for_generation获取attention_mask（仅pad为0，其余为1）
+ 这里调用_prepare_encoder_decoder_kwargs_for_generation是对于encoder-decoder模型的情况，如果没有提供encoder_outputs，这里调用encoder的forward一次，获取encoder_outputs，存入到model_kwargs中

####  **_prepare_attention_mask_for_generation**
```python
    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[torch.Tensor],
        eos_token_id: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        # No information for attention mask inference -> return default attention mask
        default_attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        if not is_input_ids:
            return default_attention_mask

        is_pad_token_in_inputs = (pad_token_id is not None) and (
            isin_mps_friendly(elements=inputs, test_elements=pad_token_id).any() 
            # isin_mps_friendly 是用来检查输入张量中是否有pad_token
        )
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(
            isin_mps_friendly(elements=eos_token_id, test_elements=pad_token_id).any()
        )
        can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = inputs.ne(pad_token_id).long()
        # ：如果 inputs 中的元素与 pad_token_id 不相等，则该位置的值为 1，表示该位置有效；否则为 0，表示该位置为填充。inputs.ne(pad_token_id) 返回一个布尔张量

        attention_mask = (
            attention_mask_from_padding * can_infer_attention_mask + default_attention_mask * ~can_infer_attention_mask
        )
        return attention_mask
```
+ 有pad则把pad的mask设为0，其余都为1

#### **generate-预处理自回归生成的input_ids**
```python
        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else: # decoder-only模型直接用之前的inputs_tensor，即(batch_sz,1)的bos_token作为input_ids
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())
```
+ _prepare_decoder_input_ids_for_generation做的事情是把input_ids或decoder_input_ids提取出来作为input_ids，如果没有start_token，再加上start_token


### Trainer 源码解析
transformers version 4.46.1
#### **初始化参数**
```python

class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.

            </Tip>

        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `processing_class` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or tokenizer.
        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
            This supercedes the `tokenizer` argument, which is now deprecated.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, here is one using
            the loss function from `transformers`
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """
```
参数如下：
- model: PretrainedModel or torch.nn.Module，如非huggingface model，需要提供model_init函数
- args: TrainingArguments
- data_collator: 定义如何将单个样本组装成一个batch，如可使用DataCollatorWithPadding来处理动态长度的输入
- train_dataset: torch.utils.data.Dataset或datasets.Dataset
- eval_dataset: 和train_dataset类似
- processing_class: 预处理类，包括`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`，怎么用待进一步了解
- model_init: Callable，每次`trainer.train()`时调用，用于自定义初始化模型方法
- compute_loss_func: Callable，接收整个accumulated batch，包括模型原始输出（如logits）、labels等，计算并返回loss(一个单元tensor，包含gradient，用于之后backward)，需要自定义loss时使用
- compute_metrics: Callable, 接收EvalPrediction参数，里面包含处理过后的模型输出、labels等，用于eval metrics的计算。当args中的batch_eval_metrics设为True时，还额外接收一个compute_result参数，用于指示当前是否需要计算，只有在eval batch的最后才会触发计算
- callbacks: 自定义callback，没用过，不清楚
- optimizers: 一个tuple，包含torch.optim.Optimizer和torch.optim.lr_scheduler.LambdaLR，默认为AdamW和用args中的参数调用get_linear_schedule_with_warmup初始化一个lr_scheduler
- preprocess_logits_for_metrics: Callable，接收模型输出的logits和数据的labels，在传给compute_metrics计算前进行预处理，其返回值将出现在EvalPrediction.predictions中。此外，可以自定义此函数用于解决eval阶段爆显存的问题，见[Eval-OOM](/posts/notes/#huggingface-tranier-oom-during-evaluation)

官方指出的一些重要属性:
- model: 指向核心模型，如PretrainedModel
- model_wrapped: 如果被Deepspeed等包装，则指向最外层的模型
- is_model_parallel: 是否使用模型并行（tensor parallel）
- place_model_on_device: 控制模型是否自动放到指定device上，如果用了Deepspeed或model parallel，将为False
- is_in_train: 模型是否在训练过程中（evaluate也算）

#### **trainer.train()**
```python
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
```
初始化参数
- resume_from_checkpoint: 指定是否从checkpoint开始恢复训练
- trial: 超参数搜索相关设置，没用过，不了解
- ignore_keys_for_eval: eval时忽略模型输出的某些key（如果输出形式是dict类型的话）

```python
    # from GPT-4o ❤
    if resume_from_checkpoint is False:
        resume_from_checkpoint = None  # 如果布尔值为False，则不恢复检查点

    # 初始化内存监控
    self._memory_tracker.start()

    args = self.args  # 获取训练参数配置

    self.is_in_train = True  # 标记为训练状态

    # 如果启用了NEFTune噪声增强，则激活模型的相关钩子
    if self.neftune_noise_alpha is not None:
        self.model = self._activate_neftune(self.model)

    # 确保在某些情况下将模型移至设备上（如启用了特定评估模式但未训练时）
    if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train and not self.is_model_parallel:
        self._move_model_to_device(self.model, args.device)

    # 处理已弃用的`model_path`参数
    if "model_path" in kwargs:
        resume_from_checkpoint = kwargs.pop("model_path")  # 将其映射到resume_from_checkpoint
        warnings.warn(
            "`model_path`已弃用，未来版本将移除。请改用`resume_from_checkpoint`。",
            FutureWarning,
        )

    # 检查是否存在未预料的关键字参数，抛出异常
    if len(kwargs) > 0:
        raise TypeError(f"train()收到未预期的关键字参数: {', '.join(list(kwargs.keys()))}.")

    # 设置超参数搜索环境（包括随机种子等）
    self._hp_search_setup(trial)
    self._train_batch_size = self.args.train_batch_size  # 初始化训练批量大小

    # 模型重新初始化
    model_reloaded = False
    if self.model_init is not None:
        # 设置随机种子以确保模型初始化的确定性
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.model = self.call_model_init(trial)  # 根据超参数重新初始化模型
        model_reloaded = True
        # 同时重新初始化优化器和学习率调度器
        self.optimizer, self.lr_scheduler = None, None

    # 如果启用了检查点恢复
    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        resume_from_checkpoint = get_last_checkpoint(args.output_dir)  # 获取最后的检查点
        if resume_from_checkpoint is None:
            raise ValueError(f"输出目录({args.output_dir})中未找到有效的检查点")

    if resume_from_checkpoint is not None:
        # 在没有特定分布式训练启用时，直接从检查点加载模型状态
        if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint)
        # 更新批量大小（如果适用）
        state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        if state.train_batch_size is not None:
            self._train_batch_size = state.train_batch_size

    # 如果模型重新初始化，将其移动到目标设备并更新包装模型
    if model_reloaded:
        if self.place_model_on_device:
            self._move_model_to_device(self.model, args.device)
        self.model_wrapped = self.model  # 更新模型包装对象

    # 查找适合的批量大小并运行内部训练循环
    inner_training_loop = find_executable_batch_size(
        self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
    )
    if args.push_to_hub:  # 如果启用了将模型推送到Hugging Face Hub
        try:
            hf_hub_utils.disable_progress_bars()  # 临时禁用进度条
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        finally:
            hf_hub_utils.enable_progress_bars()  # 确保训练完成后重新启用进度条
    else:
        # 普通情况下运行内部训练循环
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

```

#### **trainer._inner_training_loop()**
训练流程代码


<!-- ### Q
+ attention_mask 怎么设置（decoder）
+ positional embedding
+ k-v cache concat 问题
+ 为什么attention要过一层softmax
  + 部分解决：将根据QK得到的注意力分数转化为注意力权重
+ dropout作用  
+ ffn作用，为什么要放大再投回来
+ word_embedding
+ how to generate  (https://huggingface.co/blog/how-to-generate)
+ generation_config -->

### Positional Embedding
#### **位置编码**
Transformer中的自注意力机制无法捕捉位置信息，这是因为其计算过程具有置换不变性(permutation invariant)，导致打乱输入序列的顺序对输出结果不会产生任何影响。


[ref](https://0809zheng.github.io/2022/07/01/posencode.html)

### Questions
ref: [深度学习自然语言处理](https://github.com/DA-southampton/NLP_ability/tree/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)
#### **为什么使用多头注意力 (to do: dim error)**
使用多头注意力可以学习到不同的注意力权重，关注到不同的子空间，可以更好地获取输入序列中不同位置的关系信息。

具体来说，看attention计算公式：

假设

$x$: (1,seq_len, hidden_sz), $x_q$: (1,query_len, hidden_sz), n=num_head;

$W_{ki},W_{vi}$: (hidden_sz , head_dim),$ W_{qi}$: (hidden_sz,head_dim); 

$W_k,W_v$: (hidden_sz, n* head_dim=hidden_sz),  $W_q = (W_{q1},W_{q2},..,W{q_n})$: (hidden_sz,hidden_size); 

$$
\begin{align*}
\textbf{each head:} Attn_i &= softmax(\frac{x_q W_{qi} W_{ki}^T x^T}{\sqrt{\text{head_dim}}})W_{vi} x \quad\text{  :(1, query_len, head_dim)} \\
Attn = (Attn_i)_n &= (softmax(\frac{x_q W_{q1} W_{k1}^T x^T}{\sqrt{\text{head_dim}}})W_{v1}x,...,softmax(\frac{x_q W_{qn} W_{kn}^T x^T}{\sqrt{\text{head_dim}}})W_{vn}x) \quad\text{  :(1, query_len, n* head_dim=hidden_sz)}
\end{align*}
$$
$$
\begin{align*}
\textbf{no head}: Attn &= softmax(\frac{x_q W_q W_k^T x^T}{\sqrt{\text{hidden_sz}}}) W_v x \\
& = softmax(\frac{\sum_{i=1}^n x_q W_{qi} W_{ki}^T x^T}{\sqrt{\text{hidden_sz}}}) W_v x\\
& = (softmax(\frac{\sum_{i=1}^n x_q W_{qi} W_{ki}^T x^T}{\sqrt{\text{hidden_sz}}}) W_{v1} x, ..., softmax(\frac{\sum_{i=1}^n x_q W_{qi} W_{ki}^T x^T}{\sqrt{\text{hidden_sz}}}) W_{vn} x) \quad\text{  :(1, query_len, hidden_sz)}
\end{align*}
$$
多头注意力的情况下，每个头的注意力权重是不同的，因此可以关注到不同子空间的信息；单头注意力情况下虽然最终输出维度相同，但把hidden_sz维度按照num_head*head_dim方式切分后可以发现同一个注意力权重重复了num_head次

#### **self-attention 为什么Q,K,V使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？**
可以在不同空间进行投影，提取到更多信息。相同权重模型可能无法很好区分Q,K,V

#### **计算attention时为何选择点乘而不是加法？两者在计算复杂度和效果上有什么区别？**
点乘可以通过矩阵乘法的方式进行并行计算优化，比矩阵加法的并行化实现更容易更高效。理论复杂度一样，但实际上加法之后的非线性激活函数较难并行，因此效果更差。

从数学角度看，点乘是一种衡量两个向量相似度的自然方式。当查询 Q 和键 K 的方向相似时，点积值会较大，softmax后的权重也会较大，意味着这种相似性直接影响了注意力权重的大小。这种直接使用相似性进行权重分配的方式非常直观且高效。

#### **为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解**
[ref](https://blog.csdn.net/ytusdc/article/details/121622205)

#### **在计算attention score的时候如何对padding做mask操作？**
根据attetnion mask的标记，将不被注意的位置（mask为0）的值都设为较大的负值(如-100)，这样经过softmax之后几乎就基本等于0，也就不会计算该位置的attention score
(to do: code analysis)

#### **简单讲一下Transformer中的残差结构以及意义.**
残差结构广泛认为由ResNet引入，主要作用为解决梯度消失和权重矩阵退化的问题。

梯度消失是因为根据链式法则，梯度是相乘的，一旦某些项梯度很小，深度网络连乘之后整个梯度会变得非常小。加上残差结构使得每项梯度变为(1+grad)，避免了梯度消失。

权重矩阵退化是因为虽然梯度范数大，但是如果网络的可用自由度对这些范数的贡献非常不均衡，也就是每个层中只有少量的隐藏单元对不同的输入改变它们的激活值，而大部分隐藏单元对不同的输入都是相同的反应，此时整个权重矩阵的秩不高。并且随着网络层数的增加，连乘后使得整个秩变的更低。虽然是一个很高维的矩阵，但是大部分维度却没有信息，表达能力没有看起来那么强大。残差连接正是强制打破了网络的对称性，一定程度上缓解了矩阵低秩的问题，提升了网络的表征能力。

[ref](https://zhuanlan.zhihu.com/p/42833949)

#### **为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？**
Transformer使用LayerNorm而非BatchNorm是因为LayerNorm对每个样本独立进行归一化，适合变长输入序列的处理。而BatchNorm在序列建模中会受到批次大小和序列长度的影响，导致不稳定。

LayerNorm通常放置在每个子层的输出之后。即attention和feed forward之后

#### **Encoder端和Decoder端是如何进行交互的？**
在decoder的cross-attention模块进行交互，encoder最后输出的hidden_states作为cross-attention的key和value，decoder的self-attention模块输出的decoder_hidden_states作为query，进行cross-attention实现交互

#### **Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？**
Decoder阶段的多头自注意力需要进行序列mask操作，以防止模型在生成当前词时查看未来的词。而Encoder的多头自注意力则不需要mask，因为它可以同时看到输入序列的所有信息。

#### **Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？**
Transformer的并行化主要体现在Encoder的多个层和多头注意力机制的并行计算上。Decoder端在生成序列时，由于需要依赖前一个时间步的输出，通常难以完全并行化，但在Decoder的每层内部仍可以进行并行处理。



## Megatron
### 数据并行
在每个worker之上复制一份模型，这样每个worker都有一个完整模型的副本。输入数据集是分片的，一个训练的小批量数据将在多个worker之间分割；worker定期汇总它们的梯度，以确保所有worker看到一个一致的权重版本。对于无法放进单个worker的大型模型，人们可以在模型之中较小的分片上使用数据并行。

数据并行扩展通常效果很好，但有两个限制：

a）超过某一个点之后，每个GPU的batch size变得太小，这降低了GPU的利用率，增加了通信成本；

b）可使用的最大设备数就是batch size，着限制了可用于训练的加速器数量。

同一个Data Parallel Group内的数据是不同的，相当于把整个输入数据切分成dp size个DP group

### 模型并行
模型并行模式会让一个模型的内存和计算分布在多个worker之间，以此来解决一个模型在一张卡上无法容纳的问题，其解决方法是把模型放到多个设备之上。

模型并行分为两种：流水线并行和张量并行，就是把模型切分的方式。

流水线并行（pipeline model parallel）是把模型不同的层放到不同设备之上，比如前面几层放到一个设备之上，中间几层放到另外一个设备上，最后几层放到第三个设备之上。

张量并行则是层内分割，把某一个层做切分，放置到不同设备之上，也可以理解为把矩阵运算分配到不同的设备之上，比如把某个矩阵乘法切分成为多个矩阵乘法放到不同设备之上。

#### **通信**
我们接下来看看模型并行的通信状况。

张量并行：通信发生在每层的前向传播和后向传播过程之中，通信类型是all-reduce，不但单次通信数据量大，并且通信频繁(一次forward+backward需要4次all-reduce)。

流水线并行：通信在流水线阶段相邻的切分点之上，通信类型是P2P通信，单次通信数据量较少但是比较频繁，而且因为流水线的特点，会产生GPU空闲时间，这里称为流水线气泡（Bubble）。

因为张量并行一般都在同一个机器之上，所以通过 NVLink 来进行加速，对于流水线并行，一般通过 Infiniband 交换机进行连接。

#### **MLP（feedforward）部分切分方法**
切分方法如图所示
![megatron-mlp-parallel](images/notes/megatron-mlp-parallel.png)
假设Y=ACT(XA)，如果A沿行切，那么需要X沿列切，最终得到Y=ACT(X1A1+X2A2)，由于ACT的非线性，这里Y不等于ACT(X1A1)+ACT(X2A2)，因此需要reduce一次才能计算Y，没法并行

但如果A沿列切，则Y=ACT(XA1,XA2)，ACT作用于最后一维hidden_sz的每个元素上，这样通过并行后拼接可以实现激活函数的并行，因此需要将权重函数沿列切（即沿最后一维切）

X:(bz,seq_len,hidden_sz), A:(hidden_sz,ffn_hidden_sz), Ai:(hidden_sz,ffn_hidden_sz_i)

这是第一个Linear+激活函数的并行方法，上一步并行分别在两个GPU上得到Y=(Y1,Y2)，下一步需要经过另一个线性层，Z=DROPOUT(YB),刚好Y是列切，那么将B行切成B1和B2即可,
Z=DROPOUT(Y1B1+Y2B2),在这里做reduce得到输出Z

#### **self-attention部分切分方法**
直接按注意力头切即可

#### **梯度传导**
(to do: more)
矩阵求导分割转化

### 并行配置
#### **参数解释**
+ p: pplp size
+ t: tp size
+ d: dp size
+ n: num of gpus = p * t * d
+ B: global batch size
+ b: micro batch size
+ m $=\frac{B}{b *  d}$ num of microbatches per ppl，当m为1时，相当于B=b*d，即对global batch数据按d进行切分，每个dp组内的micro batch size为B/d



#### **Example**
+ n = 16 = {node1:0-7,node2:8-15}
+ tp = 2
+ pp = 4

分组为
TP: group size为2，共8个组: [0,1],[2,3],[4,5],...,[14,15]，每个group表示一组张量并行，tp的通信仅在组内进行
PP: group size为4，共4个组: [0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15], 每个group表示一组流水线并行，pp优先机间进行，pp的p2p通信仅在组内完成
DP: n/(tp*pp) = 2，说明复制了两个模型，dp为2，group size为2，共8个组: [0,2],[1,3],[4,6],[5,7],...,[13,15]，每个group表示一组数据并行，组内各GPU的数据不同，一个组的数据合并之后为global data

## Deepspeed
### 显存占用
+ 假设模型参数量为M，数据格式为fp16，则现存为2M Bytes。 
+ 每个参数对应一个梯度，所以梯度为2M
+ Adam优化器
  + 包含fp32的参数备份，大小为4M
  + fp32的一阶momentum，大小为4M
  + fp32的二阶variance，大小为4M
+ 总计16M Bytes
+ 此即所谓混合精度训练

### Zero Stages
+ Baseline
    + 每个GPU上存参数+梯度+优化器参数
    + 每卡16M
    + 每个step后进行一次all-reduce计算梯度均值，根据[环状通信](https://zhuanlan.zhihu.com/p/504957661)，对每张卡而言，发送加接收的总通信数据量近似为2M
+ Stage 1
    + 对优化器参数进行切分，N个GPU每个GPU保存1/N的优化器状态量，合并一起成为一个总的优化器状态量
    + 每卡4M+12M/N
    + 通信量同stage2
+ Stage 2
    + 对模型梯度进行切分，每个GPU保存1/N梯度
    + 每卡2M + 14M/n
    + 每卡计算1/N梯度均值，需要一次reduce，通信量M; 算完梯度更新优化器状态，需要一次gather,通信录M
+ Stage 3
    + 对参数再进行切分，每个GPU保存1/N参数
    + 每卡16M/n
    + 多了tp的通信

### Zero-Offload
四种计算节点: FWD,BWD,Param Update和float2haf。

FWD和BWD放在GPU，后两个放在CPU计算

多卡场景的offload需要Stage2：一个CPU进程对应一个GPU，负责1/N的梯度和优化器状态。但是GPU和CPU通信总量是恒定的，只和参数量有关，和N无关。


 
## vLLM
https://mp.weixin.qq.com/s/-5EniAmFf1v9RdxI5-CwiQ





