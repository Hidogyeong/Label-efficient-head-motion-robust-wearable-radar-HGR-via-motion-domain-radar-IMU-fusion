# Experiment Plan for Supervisor Feedback

## Motivation

Supervisor feedback indicates that the previous Table 5 was only a backbone comparison. If the paper claims a new training/fusion protocol, it must compare against existing fusion/training methods.

## Proposed new experiment axis

### Fusion/training strategies

1. Radar-only
2. Early fusion
3. Late fusion
4. Attention fusion
5. Motion-domain proposed

### Backbones

1. TCN/CNN
2. GRU
3. Transformer

## Minimal required experiments

### Stage 1: Fix backbone to TCN/CNN

Compare all 5 fusion strategies.

Purpose: Determine whether motion-domain proposed actually improves over existing fusion methods.

### Stage 2: Backbone robustness

Compare radar-only, early fusion, and motion-domain proposed across TCN/CNN, GRU, Transformer.

Purpose: Verify that the conclusion is not an artifact of a single backbone.

### Stage 3: Label-efficiency

Compare early fusion and motion-domain proposed at 0/10/25/50/100% non-static head labels.

Purpose: Verify that the proposed strategy is still useful under limited moving-head labels.

## Decision rule

- If motion_domain_proposed is best or competitive across Stage 1 and Stage 2, the paper can remain a fusion/training protocol paper.
- If early_fusion, late_fusion, or attention_fusion is consistently better, the paper should be reframed as an empirical benchmark and analysis paper.
