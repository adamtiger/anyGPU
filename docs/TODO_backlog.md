# Tasks may be addressed later

- [ ] mamba layer (implementation)
    - [link](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/models/zamba2/mamba2_layer.py#L36)
	- [x] conv1d (grouped, padded, dilation?)
	    - [causal-conv1d github](https://github.com/Dao-AILab/causal-conv1d/tree/main)
		- guess: y = self.silu(self.conv1d(x)) where:
        - conv1d:  nn.Conv1d(
		    in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=self.d_conv - 1)[..., :seqlen]
	- [x] implement [RMSnormGated](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py#L45) 
        - guess: z * sigmoid(z) * rms_norm(x) or rms_norm(x * z * sigmoid(z)) ; depends on parameters
	    - **can require group based mean calculation!**, but in the case of the zamba2 it seems it falls back to normal rmsnorm
	
	- [?] implement [mamba_chunk_scan_combined](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py#L281)
	    - reference implementations are also available for understanding the big picture
		- required fixing their implementation
		- further more, extensive scan is still required to see if it can work properly
		- [mamba ssm description](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546)
		- it seems the reference implementation is correct, small deviations are present but overall, 
		    the results are fine


