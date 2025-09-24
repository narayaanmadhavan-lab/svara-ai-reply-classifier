### If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

I would use data augmentation (paraphrasing with back-translation or small synthetic edits), semi-supervised learning (train on unlabeled replies with pseudo-labeling and confidence thresholds), and active learning to label only the most informative examples. I would also use transfer learning (fine-tune a pre-trained transformer) and strong cross-validation to make the most of limited labels.

### How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?

I would audit training data for demographic or company-specific bias, remove sensitive attributes, and add label and performance monitoring for disparate error rates across groups. Deploy with a human-in-the-loop for low-confidence or high-impact predictions and add guardrails (e.g., don't act on "positive" unless confidence > threshold), plus periodic re-training.

### Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

Provide a concise context (recipient role, company, recent event), a required tone and length, and a few positive/negative examples (few-shot). Constrain the output format (one-liner, 15 words) and include variables to be replaced, plus an instruction to avoid generic phrases and to include one specific detail (e.g., recent product launch).
