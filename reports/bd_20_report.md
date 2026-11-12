# Model Security Report: bd_20
**Verdict:** SUSPICIOUS - MANUAL REVIEW  
**P(backdoor):** 0.49 (UNCALIBRATED heuristic (run make_zoo.py first))  
**Suspected target class:** 7  
**Clean accuracy on scan set:** 71.2%

## Evidence
- quiet - Trigger search (Neural Cleanse): class 7: mask-size anomaly index 0.96 (>2 suspicious); using only 3% of pixels its trigger still flips 88% of fresh clean inputs vs 64% for the median class; mask is 0.70x the median class mask
- quiet - Behavioural: random-patch hijack: a single random patch pushed 44% of inputs into one class
- **FIRED** - Behavioural: STRIP entropy test: under blending, triggered inputs keep 87% lower entropy than clean ones (trigger survives blending)
- **FIRED** - Activation: neuron elevation (ABS-style): one label is driven by a single channel with outlier elevation score 5.94
- **FIRED** - Activation: spectral signature / clustering: triggered inputs separate from genuine target inputs (spectral AUC 1.00, cluster agreement 0.98)

![trigger](bd_20_trigger.png)