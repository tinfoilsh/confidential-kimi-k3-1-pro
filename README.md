# Confidential Kimi K3.1 Pro

Confidential serving configuration for `moonshotai/Kimi-K3.1-Pro` with vLLM.

Uses the Inferact DSpark draft, FP8 KV cache and FP32 recurrent state. Model
revisions and verified model packs are pinned in `tinfoil-config.yml`.

Run the **Tinfoil Release** workflow with a version to build the runtime image
and publish the signed deployment configuration. The release workflow replaces
the image digest placeholder in the configuration.
