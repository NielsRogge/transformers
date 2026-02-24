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



### Update 15 (current)

- Extended query-stage parity coverage to include both 2-frame and 3-frame inputs, still comparing 5D video tensors against flattened 4D frame batches at every layer output.
- Updated both conversion-time validation and targeted modeling tests to run this all-layer parity check across multiple temporal lengths.
- **Status:** still in backbone/query-stage verification; no new head architecture changes were implemented in this update.

## Implemented in this update

- [x] Milestone 1 (mask-layout support + 4D/5D embedding consistency checks, masked and unmasked).
- [x] Milestone 2 (model-level 5D input adaptation baseline).
- [ ] Milestone 3+
