# Model Security Report: vit-base-patch16-224-in21k-finetuned-cifar10
**Verdict:** NO STRONG EVIDENCE OF BACKDOOR  
**P(backdoor):** 0.02 (calibrated ensemble)  
**Suspected target class:** 1  
**Clean accuracy on scan set:** n/a (reference labels do not match model classes)

## Evidence
- quiet - Trigger search (Neural Cleanse): class 1: mask-size anomaly index 5.84 (>2 suspicious); using only 3% of pixels its trigger still flips 2% of fresh clean inputs vs 1% for the median class; mask is 0.96x the median class mask
- quiet - Behavioural: random-patch hijack: a single random patch pushed 6% of inputs into one class
- quiet - Behavioural: STRIP entropy test: under blending, triggered inputs keep 0% lower entropy than clean ones (trigger survives blending)
- quiet - Activation: neuron elevation (ABS-style): one label is driven by a single channel with outlier elevation score 0.92
- quiet - Activation: spectral signature / clustering: triggered inputs separate from genuine target inputs (spectral AUC 0.99, cluster agreement 0.73)
- quiet - Transformer: MLP-unit elevation (informational): clamping one hidden unit of the last block MLP drives label 9 with outlier elevation score 0.92 (>4 suspicious)
- quiet - Transformer: attention on reconstructed trigger (diagnostic, not validated): CLS attention on the 6/196 patches under the reconstructed trigger is 0.4x uniform, 0.5x its value on clean images, 0.4x a random patch there; strongest single head 1.2x clean

![trigger](vit-base-patch16-224-in21k-finetuned-cifar10_trigger.png)