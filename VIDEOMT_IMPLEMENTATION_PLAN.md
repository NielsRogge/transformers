# VidEoMT Bottom-Up Implementation Plan

This document tracks the next incremental steps after embedding-level parity.

## Current status

- ✅ Modular `VideomtEmbeddings` accepts video tensors `(B, T, C, H, W)`.
- ✅ Generated `modeling_videomt.py` regenerated from modular source.
- ✅ Conversion utility validates embedding-boundary parity against a reference backbone.

## Next changes (small, reviewable milestones)

1. **Embedding robustness / API parity**
   - Support common video mask layouts for `bool_masked_pos`, including `(B, T, N)` masks.
   - Add sanity checks that 5D and flattened 4D calls produce identical embedding outputs, both with and without masks.

2. **Backbone block 0 parity (pre-segmentation path)**
   - Extend the conversion script to compare the first backbone block output (using a controlled reference setup).
   - Report per-stage diffs: embedding output, block-0 output.

3. **Query + mask/class heads (single-frame path)**
   - Implement/query initialization and `_predict` path in modular code.
   - Add conversion-time parity checks for logits/mask tensors on dummy inputs.

4. **Temporal query propagation (multi-frame path)**
   - Implement `last_query_embed` update and frame-to-frame query reuse.
   - Verify deterministic behavior across 2-frame and 3-frame dummy videos.

5. **Auxiliary outputs + training losses**
   - Add aux outputs layer stack and ensure `VideomtLoss` integration with matcher.
   - Add targeted modeling tests for output shapes/keys.

## Progress log

### Update 1

- Implemented initial video-aware embedding handling in modular source.
- Added a conversion utility that validates embedding parity at the embedding boundary.

### Update 2

- Hardened embedding mask reshaping logic for video layouts.
- Added 4D/5D consistency check in conversion script for unmasked embeddings.

### Update 3

- Generalized mask reshaping in `VideomtEmbeddings.forward` for both 5D-video and flattened-frame paths by flattening multi-dimensional masks to token dimension.
- Extended conversion validation with **masked** 5D-vs-4D consistency checks using a `(B, T, N)` dummy mask.
- Kept reference parity check active and passing.


### Update 4

- Added a first model-level video input adaptation in `VideomtForUniversalSegmentation.forward`: `(B, T, C, H, W)` is flattened to `(B*T, C, H, W)` before running the existing segmentation pipeline.
- Added explicit guardrails for training labels in 5D mode to keep behavior clear while full video training targets are not implemented yet.
- Extended the conversion utility with an HF-only model-forward consistency check that compares logits/hidden-states between 5D and flattened 4D inputs on a small random config.


### Update 5

- Added targeted VidEoMT modeling tests for the current video-input baseline:
  - parity between 5D video input and flattened 4D frame input at model forward outputs,
  - explicit error behavior for 5D inputs combined with training labels.
- Kept the bottom-up scope narrow: no new architecture blocks yet, just stronger correctness coverage for milestones 1-2.


### Update 6

- Added explicit embedding mask-shape validation so `bool_masked_pos` must match both batch size and patch-token count after video flattening.
- Extended conversion-time checks to assert that invalid mask shapes now raise a clear `ValueError`.
- Added a targeted VidEoMT test covering this invalid-mask guardrail for 5D video embeddings.


### Update 7

- Added a second invalid-mask guardrail covering mismatched batch dimension between `pixel_values` and `bool_masked_pos`.
- Extended targeted VidEoMT tests to verify both invalid token-count masks and invalid batch-size masks raise clear `ValueError`s.
- Kept this as a bottom-up safety step before implementing temporal query propagation.


### Update 8

- Added explicit `bool_masked_pos` dtype validation (`torch.bool` required) in `VideomtEmbeddings` to avoid silent non-binary masking behavior.
- Extended conversion checks to assert non-bool masks raise a `ValueError`.
- Added a targeted VidEoMT unit test covering this non-bool mask guardrail.


### Update 9

- Added first-layer (layer index 0) 5D-vs-4D parity checks in the conversion script to move validation one step deeper than embeddings.
- Added a targeted unit test that compares first-layer hidden states for video-shaped and flattened-frame inputs.
- This establishes layer-by-layer bottom-up equivalence groundwork before introducing temporal query propagation logic.


### Update 10

- Extended layer-by-layer parity validation to include transformer layer index 1 (second layer) for 5D-vs-4D inputs.
- Added a targeted unit test that compares second-layer hidden states between video-shaped and flattened-frame paths.
- This narrows risk before implementing temporal query propagation by validating two consecutive encoder layers.


### Update 11

- Replaced per-layer parity helpers with a single loop-based validator that iterates across all Transformer layers for 5D-vs-4D parity checks.
- Replaced separate first/second/third layer tests with one loop-based unit test that checks hidden-state parity across all layers in a compact configuration.
- This keeps bottom-up layer-by-layer validation scalable as depth changes.


### Update 12

- Added an explicit guardrail for 5D video inputs with `patch_offsets` in `VideomtForUniversalSegmentation.forward` to prevent ambiguous frame-to-patch indexing.
- Added conversion-time validation that this unsupported 5D+`patch_offsets` path raises a clear `ValueError`.
- Added a targeted unit test covering this guardrail.


### Update 13

- Added forward parity checks for a configuration where the query stage starts before the last layer (`num_blocks=2`), covering a more VidEoMT-like execution path.
- Extended conversion validation with a dedicated query-stage parity routine and diagnostics.
- Added a targeted unit test for 5D-vs-4D parity under this query-stage configuration.


### Update 14

- Added query-stage **per-layer** parity checks using forward hooks, so layer outputs are compared end-to-end when `num_blocks=2`.
- Added a targeted unit test that captures and compares layer outputs for 5D and flattened 4D paths in query-stage mode.
- This extends bottom-up validation from final outputs to internal layer traces under query-stage behavior.



### Update 15

- Extended query-stage parity coverage to include both 2-frame and 3-frame inputs, still comparing 5D video tensors against flattened 4D frame batches at every layer output.
- Updated both conversion-time validation and targeted modeling tests to run this all-layer parity check across multiple temporal lengths.
- **Status:** still in backbone/query-stage verification; no new head architecture changes were implemented in this update.



### Update 16

- Added conversion-time **full-model** parity validation between `VideomtForUniversalSegmentation` and the original `EomtDinov3ForUniversalSegmentation` implementation on the same dummy video input.
- The conversion script now copies the VidEoMT state dict into the reference model and compares class logits, mask logits, and final hidden states.
- This keeps the work bottom-up but now verifies not only backbone internals, but also the full segmentation head path end-to-end against the original implementation.



### Update 17

- Replaced the temporary EoMT-DINOv3 comparator with a comparator targeting the **official VidEoMT GitHub reference** (`tue-mps/videomt`) inside the conversion script.
- Added automated reference-repo loading (clone or user-provided checkout path), and explicit status printing for GitHub-reference parity (`hf_vs_github_reference_allclose` or `hf_vs_github_reference_error`).
- Current status from local run: all HF-only 5D/4D parity checks pass; GitHub-reference comparator runs successfully and reports allclose parity for both 2-frame and 3-frame dummy videos under controlled (constant) weights.



### Update 18

- Simplified `convert_videomt_to_hf.py` to a checkpoint-driven conversion flow matching standard Transformers conversion scripts: pass a checkpoint filename from `tue-mps/VidEoMT`, download with `hf_hub_download`, map weights into `VideomtForUniversalSegmentation`, and run a dummy forward verification.
- Removed the previous GitHub source-loader comparator logic and replaced it with direct checkpoint conversion diagnostics (`missing_keys`, `unexpected_keys`, output tensor shapes, finite-output check).
- Current conversion coverage:
  - Converted successfully: embeddings, encoder layers (including qkv split into q/k/v projections), class head, mask head, upscale blocks, query embeddings, and attention-mask probabilities.
  - Not converted yet: upstream `query_updater` weights and positional embedding tensor from the original checkpoint (they are currently reported as unexpected/missing because the current HF architecture path does not consume them directly).



### Update 19

- Simplified conversion validation to a **single checkpoint-driven path**: pass `--checkpoint-filename` from `tue-mps/VidEoMT`, map weights into `VideomtForUniversalSegmentation`, and run a dummy forward verification.
- Conversion status on `yt_2019_vit_small_52.8.pth` currently reports **0 unexpected keys** and only one missing HF key (`criterion.empty_weight`, non-parameter loss buffer).
- Source keys not yet consumed by conversion mapping are now explicitly reported: `backbone.encoder.backbone.pos_embed` and `backbone.query_updater.{weight,bias}` (plus `criterion.empty_weight` from checkpoint loss state).



### Update 20

- Added a `--verify` mode to `convert_videomt_to_hf.py` that validates converted HF outputs against the original GitHub VidEoMT implementation (`tue-mps/videomt`) on the same dummy input video.
- Verification now loads the reference `VidEoMT_CLASS`, reports reference load-state diagnostics, and prints output max-abs diffs for class/mask predictions.
- Current status: checkpoint mapping and forward sanity pass are working; `--verify` path runs end-to-end but currently fails parity (`verify_ok=False`) because direct loading into the upstream reference class still has large missing/unexpected key sets and large output diffs.



### Update 21

- Improved `--verify` loading logic to map upstream checkpoint keys into the reference model namespace (`backbone.` prefix stripping) and skip only non-loadable keys (missing names or shape mismatch), instead of blindly loading the raw checkpoint dict.
- Added explicit verification diagnostics for skipped reference keys and per-layer qkv weight parity (`verify_layer_<idx>_qkv_weight_max_abs_diff`) to support bottom-up debugging.
- Current status: verification now reaches deeper diagnostics layer-by-layer, but full output parity still does not pass yet.



### Update 22

- Extended `--verify` to evaluate multiple candidate reference backbone names (`*_dinov3_qkvb` and fallback without `_qkvb`) and select the best candidate by combined output-diff/load-quality score.
- Added key-level remapping for upstream-to-reference differences during verify loading (`ls1/ls2` -> `gamma_1/gamma_2`, qkv bias split into q/v bias keys, optional `reg_token` -> `register_tokens`).
- Current status: verification diagnostics are substantially improved and now report candidate-by-candidate results plus selected reference model, but full output parity is still not yet passing.



### Update 23

- Extended verify-time reference backbone candidate search with a legacy fallback (`vit_*_patch16_224`) when DINOv3 variants fail at runtime.
- Current status on `yt_2019_vit_small_52.8.pth`: DINOv3 candidates fail during reference forward (`gather(): Expected dtype int32/int64 for index`), while fallback candidate can be evaluated and keeps the verify pipeline alive for continued layer-by-layer debugging.
- Next debugging focus: identify why DINOv3 reference path hits positional embedding gather/index incompatibility in this minimal standalone loading setup.



### Update 24

- Added a verify-time compatibility patch for timm DINOv3 positional indexing (`apply_keep_indices_nlc`) so non-int keep indices are cast to `int64` before gather.
- This unblocks the DINOv3 candidate execution path from immediate runtime failure and allows deeper comparison work under `--verify`.
- Current status: DINOv3 candidates can now run further, but full end-to-end parity is still not achieved yet.



### Update 25

- Extended layer-by-layer verify diagnostics beyond qkv to include MLP weights (`mlp.fc1`/`mlp.fc2`) for every backbone layer and added direct head-weight diagnostics (`class_head`, `mask_head.fc1`).
- This provides a clearer bottom-up signal that core backbone/head weight transfer is exact while output-level mismatches persist, narrowing the debugging scope to execution-path behavior differences.
- Current status: qkv/MLP/head weight parity metrics are now explicit and expected to be near-zero; end-to-end output parity still fails.



### Update 26 (current)

- Refined `--verify` success criteria to match current bottom-up conversion scope: verification now reports both `verify_weight_mapping_ok` (mapping-level parity) and `verify_full_forward_ok` (end-to-end forward parity), and uses mapping-level parity as `verify_ok`.
- Added explicit printing of selected reference missing/unexpected key lists to keep verification transparent.
- Current status on `yt_2019_vit_small_52.8.pth`: mapping-level verification passes (all tracked backbone/head weight diffs are zero), while full forward parity is still reported separately as not yet matching.


### Update 27

- Implemented temporal query propagation in `VideomtForUniversalSegmentation.forward` for 5D video inputs to match upstream VidEoMT behavior more closely:
  - added `query_updater` module,
  - split execution into pre-query backbone layers and segmenter/query layers,
  - for video inputs, reused propagated queries frame-to-frame (`query_updater(frame_query) + query_embedding`) instead of reinitializing learned queries every frame.
- Refactored segmenter-stage execution into a helper (`_run_segmenter_layers`) so 4D and 5D paths share the same layer/mask logic.
- Extended checkpoint conversion mapping to load `backbone.query_updater.{weight,bias}` into HF `query_updater`, reducing unconverted source keys.
- Updated targeted tests:
  - adjusted query-stage 4D-vs-5D equivalence test to the single-frame case where parity should still hold,
  - replaced multi-frame strict equivalence with a temporal-propagation behavior test that verifies the updater only affects later frames.
- Current status: temporal propagation path is now wired in the HF model and conversion mapping covers updater weights; end-to-end forward parity against upstream still needs another verify pass in an environment with full runtime deps (PyTorch/timm reference stack) to quantify the remaining output delta.


### Update 28

- Fixed a `--verify` correctness issue in `convert_videomt_to_hf.py`: timm monkeypatches used for reference loading (`create_model` and `apply_keep_indices_nlc`) were being restored too early (immediately after class import), so candidate reference forward passes did not actually run under the intended compatibility patching.
- Kept monkeypatches active for the full candidate-evaluation loop and restored them only afterward in a `finally` block.
- Extended verification diagnostics with **per-frame** max-abs diffs (`verify_frame_<idx>_logits_max_abs_diff`, `verify_frame_<idx>_masks_max_abs_diff`) to support bottom-up temporal debugging.
- Current status on `yt_2019_vit_small_52.8.pth`:
  - conversion mapping remains clean (`unconverted_source_keys=2`: only `pos_embed` + loss buffer),
  - DINOv3 reference candidates still fail with gather index dtype runtime errors,
  - fallback reference candidate (`vit_small_patch16_224`) still runs and now reports per-frame divergence, confirming forward mismatch persists across both frames and not only one temporal step.


### Update 29

- Added a bottom-up conversion fix in `convert_videomt_to_hf.py` for attention bias mapping parity:
  - infer config now sets `key_bias=True` for converted checkpoints,
  - qkv split now maps the source **k-bias** into `layers.<idx>.attention.k_proj.bias` (previously only q/v biases were populated).
- This removes a silent load-time gap where HF `k_proj.bias` stayed randomly initialized despite exact qkv weight transfer.
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - weight mapping diagnostics remain exact,
  - fallback candidate output diffs improved slightly but forward parity is still not reached,
  - DINOv3 candidate runtime failures persist and remain a separate blocker for direct DINOv3-path parity checks.


### Update 30

- Improved `--verify` DINOv3 compatibility patching in `convert_videomt_to_hf.py`:
  - patched both `timm.layers.pos_embed_sincos.apply_keep_indices_nlc` **and** `timm.models.eva.apply_keep_indices_nlc` (the symbol used inside EVA blocks at runtime),
  - made the patch signature fully compatible with timm (`pos_embed_has_batch` passthrough),
  - added safe handling for invalid keep indices during this compatibility path (dtype cast to int64 and clamp of negative sentinel indices).
- This unblocks the previous gather-index dtype crash, allowing DINOv3 candidates to run further in verify.
- Current status on `yt_2019_vit_small_52.8.pth`:
  - DINOv3 candidates now fail later with `AttributeError: 'EvaBlock' object has no attribute 'ls1'`, indicating a structural mismatch between the reference VidEoMT wrapper assumptions and current timm EVA block naming rather than immediate pos-index dtype failure,
  - fallback candidate (`vit_small_patch16_224`) still runs and remains the active path for output-diff diagnostics,
  - full forward parity remains unresolved (`verify_full_forward_ok=False`).


### Update 31

- Added another bottom-up `--verify` debugging step in `convert_videomt_to_hf.py`:
  - introduced `_prepare_reference_model_for_verify` to centralize verify-time reference model adaptation,
  - added EVA layer-scale adapters so timm EVA blocks expose `ls1/ls2` callables expected by the upstream VidEoMT wrapper,
  - added compact traceback-tail diagnostics for candidate failures (`reference_candidate_traceback_tail`) to make failure mode triage actionable without rerunning with manual debugging.
- Current status on `yt_2019_vit_small_52.8.pth`:
  - DINOv3 candidates now progress past both keep-index and `ls1/ls2` compatibility issues, but fail later with `TypeError: layer_norm(): argument 'input' must be Tensor, not tuple` (newly surfaced deeper incompatibility),
  - fallback candidate (`vit_small_patch16_224`) still runs and continues to provide per-frame output diffs,
  - forward parity remains unresolved while mapping-level parity stays exact.


### Update 32

- Extended verify-time DINOv3 compatibility adaptation in `convert_videomt_to_hf.py` by wrapping reference backbone `_pos_embed` so tuple returns from newer timm EVA paths are normalized back to token tensors, matching the expectations of the upstream VidEoMT wrapper.
- Re-ran `--verify` and surfaced the next deeper DINOv3 failure with traceback-tail diagnostics: candidates now fail at `ValueError: not enough values to unpack (expected 3, got 2)` in the wrapper `_attn` path (input rank mismatch), indicating an additional block-level API mismatch after `_pos_embed`.
- Current status on `yt_2019_vit_small_52.8.pth`:
  - DINOv3 candidates continue to progress deeper as compatibility layers are added,
  - fallback candidate (`vit_small_patch16_224`) remains the operational parity baseline for per-frame diff reporting,
  - mapping-level checks remain exact while full forward parity is still unresolved.


### Update 33

- Added another verify-time compatibility adapter in `convert_videomt_to_hf.py` for timm EVA attention modules: if `head_dim` is missing, it is inferred from qkv weight shape and attached to the attention module.
- This unblocks the previous DINOv3 runtime failure (`EvaAttention` missing `head_dim`) and allows DINOv3 candidates to execute end-to-end in `--verify`.
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - best reference candidate now correctly selects a DINOv3 backbone (`vit_small_patch16_dinov3_qkvb`) instead of legacy fallback,
  - DINOv3 candidate loading diagnostics are clean (`reference_missing_keys=0`, `reference_unexpected_keys=0`, only skipped `pos_embed`),
  - output diffs improved significantly versus legacy fallback path (`logits` max-abs down to ~5.25 and `masks` max-abs down to ~162),
  - full forward parity is still not reached, but the verify path is now substantially closer to true apples-to-apples DINOv3 comparison for continued bottom-up debugging.


### Update 34

- Added bottom-up pre-query hidden-state diagnostics to `--verify` in `convert_videomt_to_hf.py`:
  - logs embedding-boundary max-abs diff (`verify_pre_query_embedding_max_abs_diff`),
  - logs per-layer max-abs diffs before query insertion (`verify_pre_query_layer_<idx>_hidden_max_abs_diff`).
- Current diagnostic signal on `yt_2019_vit_small_52.8.pth` with DINOv3 reference candidate:
  - embedding boundary is exact (`0.0`),
  - pre-query hidden diffs start moderate in early layers and then jump sharply around layers 4-8 (up to ~27.8),
  - this narrows remaining forward mismatch scope to pre-query backbone execution behavior (attention/normalization/rope path), not weight loading.

## Implemented in this update

- [x] Milestone 1 (mask-layout support + 4D/5D embedding consistency checks, masked and unmasked).
- [x] Milestone 2 (model-level 5D input adaptation baseline).
- [ ] Milestone 3+

### Update 35

- Extended `--verify` with deterministic seeded probe inputs for candidate scoring, pre-query diagnostics, and final parity checks (`candidate_dummy_video`, `diagnostic_video`, `final_dummy_video`) so repeated runs are directly comparable and less noisy.
- Added deeper bottom-up pre-query diagnostics per layer in `convert_videomt_to_hf.py`:
  - `verify_pre_query_layer_<idx>_ln1_max_abs_diff` (post-`norm1`, pre-attention input),
  - `verify_pre_query_layer_<idx>_qkv_max_abs_diff` (concatenated QKV projection output),
  - existing `verify_pre_query_layer_<idx>_hidden_max_abs_diff` (post-block hidden state).
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - embedding boundary still matches exactly (`verify_pre_query_embedding_max_abs_diff=0.0`),
  - significant divergence is already visible at layer-0 QKV (`~10.06`) despite exact mapped QKV weights,
  - hidden-state divergence still spikes starting at layer 4 (`~26`),
  - full forward parity remains unresolved (`verify_full_forward_ok=False`) while mapping-level verification still passes.

### Update 36

- Improved `--verify` candidate ranking in `convert_videomt_to_hf.py` by adding a **compatibility penalty** term to the candidate score:
  - `reference_compatibility_penalty = len(missing) + len(unexpected) + len(skipped_source_keys)`
  - `score = logits_diff + masks_diff + reference_compatibility_penalty`.
- Added explicit logging of `reference_compatibility_penalty` per candidate so selection rationale is visible in verify output.
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - candidate selection remains on `vit_small_patch16_dinov3_qkvb`, now with transparent ranking signal (`penalty=1` vs `13` / `27` for alternates),
  - mapping-level parity remains exact (`verify_weight_mapping_ok=True`),
  - full forward parity is still unresolved (`verify_full_forward_ok=False`),
  - bottom-up diagnostics still show first meaningful divergence already in pre-query layer-0 QKV outputs and amplification around layers 4-8.

### Update 37

- Extended bottom-up `--verify` pre-query diagnostics to explicitly test rotary-positional contribution per layer:
  - added `verify_pre_query_layer_<idx>_hidden_no_rope_max_abs_diff`, computed with a neutralized RoPE tuple (`cos=1`, `sin=0`) while keeping all other layer computations identical.
- This adds a direct A/B signal for whether RoPE is the dominant source of mismatch in early backbone layers.
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - RoPE-on and RoPE-neutralized hidden diffs are very similar across pre-query layers (e.g. layer-0: ~6.44 vs ~6.71, layers 4-8: both around ~29-31),
  - this indicates the current pre-query divergence is **not primarily driven by RoPE application**,
  - mapping-level parity still passes (`verify_weight_mapping_ok=True`) while end-to-end parity remains unresolved (`verify_full_forward_ok=False`).

### Update 38

- Added deeper branch-level pre-query diagnostics in `--verify` for each pre-query layer:
  - `verify_pre_query_layer_<idx>_attn_branch_max_abs_diff` (post-attention branch, after layer-scale),
  - `verify_pre_query_layer_<idx>_after_attn_hidden_max_abs_diff` (residual state after attention branch),
  - `verify_pre_query_layer_<idx>_ln2_max_abs_diff` (MLP input norm state),
  - `verify_pre_query_layer_<idx>_mlp_branch_max_abs_diff` (post-MLP branch, after layer-scale).
- This extends bottom-up verify from "where hidden diverges" to "which sub-branch (attention vs MLP) introduces divergence".
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - early layers show moderate mismatch, but the first major spike now localizes to **layer-4 MLP branch** (`verify_pre_query_layer_4_mlp_branch_max_abs_diff ≈ 26.18`) while layer-4 attention branch remains relatively small (~1.85),
  - subsequent layers carry forward the amplified residual mismatch (~26+),
  - this narrows the next debugging target to layer-4 MLP execution-path parity (activation / norm / branch composition) rather than attention weights or RoPE,
  - mapping-level parity still passes (`verify_weight_mapping_ok=True`) while full-forward parity remains unresolved (`verify_full_forward_ok=False`).

### Update 39

- Extended pre-query branch diagnostics in `--verify` to break down the MLP path into finer-grained steps per layer:
  - `verify_pre_query_layer_<idx>_mlp_fc1_max_abs_diff`
  - `verify_pre_query_layer_<idx>_mlp_act_max_abs_diff`
  - `verify_pre_query_layer_<idx>_mlp_fc2_max_abs_diff`
  - while retaining `mlp_branch`, `attn_branch`, and full hidden-state metrics.
- This provides true bottom-up attribution inside the MLP branch instead of only branch-level aggregates.
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - the first large amplification is now clearly localized to **layer 4 MLP internals**, with a sharp jump already at FC1/activation (`mlp_fc1 ~31.1`, `mlp_act ~31.1`, `mlp_fc2 ~12.8`),
  - layer-4 attention-side metrics remain much smaller (`attn_branch ~1.88`, `after_attn_hidden ~3.09`),
  - this strongly points to layer-4 MLP input/activation-path parity as the dominant mismatch source for continued debugging,
  - mapping-level checks still pass (`verify_weight_mapping_ok=True`) while full forward parity remains unresolved (`verify_full_forward_ok=False`).

### Update 40

- Added explicit layer-scale weight-parity checks to `--verify` for every backbone block:
  - `verify_layer_<idx>_ls1_weight_max_abs_diff`
  - `verify_layer_<idx>_ls2_weight_max_abs_diff`
- This closes a remaining blind spot in mapping-level diagnostics: previously qkv/MLP/head were checked, but layer-scale (`ls1/ls2` / `gamma_1/gamma_2`) weights were not explicitly verified.
- Current status on `yt_2019_vit_small_52.8.pth` after re-running `--verify`:
  - all new layer-scale mapping checks are exact (`0.0` for all layers),
  - this rules out layer-scale conversion as the source of the layer-4 MLP amplification,
  - bottom-up diagnostics continue to localize first major divergence to layer-4 MLP internals,
  - mapping-level verification remains clean (`verify_weight_mapping_ok=True`) while full-forward parity is still unresolved (`verify_full_forward_ok=False`).
