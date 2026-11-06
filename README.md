# Confidential Kimi K3.1 Pro

Confidential serving configuration for `moonshotai/Kimi-K3.1-Pro` with vLLM.

Uses the Inferact DSpark draft, FP8 KV cache and FP32 recurrent state. Model
revisions and verified model packs are pinned in `tinfoil-config.yml`.
The configuration serves up to 1,048,576 tokens with TP8 and DCP8.

The API model name is `kimi-k3-1-pro`.

Run the **Tinfoil Release** workflow with a version to build the runtime image
and publish the signed deployment configuration. The release workflow updates
the pinned image reference in the configuration.
