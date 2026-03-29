# Paper Notes — Research Connections

## Core research question
> Does sentence-level miscalibration in abstractive summarization predict
> hallucination risk? And does uncertainty-aware decoding produce better-calibrated
> AND more factually grounded summaries than post-hoc temperature scaling?

---

## Direct connections to Stengel-Eskin's work

### Wang and Stengel-Eskin, "On calibrating generation with self-generated distractors" (ICLR 2026)
This is the closest paper to our work. They calibrate generation using
self-generated distractors as negative examples. Our work is complementary:
we calibrate at the sentence level using factual consistency as the
calibration signal, and compare post-hoc vs decoding-time calibration.
Key difference: their calibration is training-time, ours is inference-time.

### Stengel-Eskin et al., "Multi-agent pragmatic reasoning for better calibration" (NeurIPS 2024)
They show that pragmatic reasoning improves calibration in multi-agent settings.
Our work investigates a simpler hypothesis: does exposing the model to its own
uncertainty during decoding (entropy penalty) improve calibration?
The entropy-penalty mechanism in our uncertainty-aware decoding is a
single-agent version of their calibration-through-reasoning idea.

---

## Supporting papers

### Guo et al., "On calibration of modern neural networks" (ICML 2017)
- Introduced temperature scaling as post-hoc calibration
- Showed that modern deep networks are systematically overconfident
- Our Method A (temperature scaling) directly implements their approach
- We extend it to the generative setting at sentence level

### Maynez et al., "On faithfulness and factuality in abstractive summarization" (ACL 2020)
- Showed 30% of BART summaries contain hallucinated content on CNN/DailyMail
- Established that hallucination is systematic, not random
- Motivates our hypothesis: miscalibration and hallucination should correlate

### Dziri et al., "On the Origin of Hallucinations in Conversational Models" (NAACL 2022)
- Hallucinations correlate with model uncertainty during generation
- Directly supports our research claim
- Our ECE-hallucination correlation plot tests their finding in the
  summarization setting with explicit calibration interventions

---

## Positioning statement

> "We investigate the relationship between calibration quality and factual
> faithfulness in abstractive summarization. Unlike prior work that treats
> calibration and hallucination as separate problems, we show they are
> correlated: sentences where the model is miscalibrated (overconfident)
> are significantly more likely to be hallucinated. We further show that
> uncertainty-aware decoding — which penalizes high-entropy generation steps
> at inference time — achieves better calibration AND lower hallucination rates
> than post-hoc temperature scaling, without any additional training."
