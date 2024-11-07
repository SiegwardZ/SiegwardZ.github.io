---
layout: post
title: Notes for NLP interview
date: 2024-11-04 19:52 +0800
categories: [Study, NLP]
comment: false
math: true
tags: [notes,ai,nlp,interview]
---
ref: [深度学习自然语言处理](https://github.com/DA-southampton/NLP_ability/tree/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)
## Transformers
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
+ attention_weights经过一层softmax再乘value（attention_mask长什么样？）

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

#### generate方法定义
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

#### generate-预检查
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

#### _prepare_model_inputs方法
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

#### _maybe_initialize_input_ids_for_generation方法
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

#### generate-模型参数补全
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

####  _prepare_attention_mask_for_generation
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

#### generate-预处理自回归生成的input_ids
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


### Q
+ layer_norm 是什么，作用是什么（encoder两次）
+ attention_mask 怎么设置（decoder）
+ positional embedding
+ k-v cache concat 问题
+ 为什么attention要过一层softmax
  + 部分解决：将根据QK得到的注意力分数转化为注意力权重
+ dropout作用  
+ ffn作用，为什么要放大再投回来
+ word_embedding
+ how to generate  (https://huggingface.co/blog/how-to-generate)
+ generation_config
+ 
