# HEROS-LLM v2 Metric Memo

This memo documents how the experiment's metrics align with prior LLM/XAI evaluation constructs and where the project intentionally uses task-specific names.

## Evaluation Framing

Following Doshi-Velez and Kim, the evaluation stack is intentionally split across two levels:

- `Functionally grounded` evaluation: objective metrics computed directly from the structured rule packet, such as `Evidence Grounding Precision`, `Key Evidence Coverage`, `Prediction-Explanation Agreement`, `Rule Coverage`, `Conflict Acknowledgment Score`, and `Confidence Wording Calibration`.
- `Human-grounded` evaluation: audience-facing quality judgments such as `Audience Understandability` and `Audience Technical Fit`, implemented here with an LLM judge as a scalable proxy rather than a replacement for human evaluation.

This separation matters because the project is not trying to evaluate free-form explanation quality in the abstract. It is evaluating whether audience-specific natural-language explanations stay faithful to structured model evidence while remaining understandable at the intended technical level.

## Anchor References

- Doshi-Velez and Kim, *Towards A Rigorous Science of Interpretable Machine Learning* (2017/2018): [arXiv](https://arxiv.org/abs/1702.08608)
- Kim et al., *Human-centered evaluation of explainable AI applications: a systematic review* (2024): [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1456486/full)
- Atanasova et al., *Faithfulness Tests for Natural Language Explanations* (ACL 2023): [ACL Anthology](https://aclanthology.org/2023.acl-short.25/)
- Parcalabescu and Frank, *On Measuring Faithfulness or Self-consistency of Natural Language Explanations* (ACL 2024): [ACL Anthology](https://aclanthology.org/2024.acl-long.329/)
- Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* (EMNLP 2023): [ACL Anthology](https://aclanthology.org/2023.emnlp-main.153/)
- Huang et al., *An Empirical Study of LLM-as-a-Judge for LLM Evaluation* (Findings ACL 2025): [ACL Anthology](https://aclanthology.org/2025.findings-acl.306/)
- Bang et al., *HalluLens: LLM Hallucination Benchmark* (ACL 2025): [ACL Anthology](https://aclanthology.org/2025.acl-long.1176/)
- Lertvittayakumjorn et al., *Diagnostics-Guided Explanation Generation* (AAAI 2022): [AAAI PDF](https://cdn.aaai.org/ojs/21287/21287-13-25300-1-2-20220628.pdf)
- Xing et al., *Towards Faithful Model Explanation in Natural Language with Evidence Attribution* (NAACL 2025): [ACL Anthology](https://aclanthology.org/2025.naacl-long.282/)

## Mapping

| Current v2 name | Prior construct | Status | Why we use it here |
| --- | --- | --- | --- |
| Evidence Grounding Precision | groundedness / evidence attribution / faithfulness-to-input | Literature-aligned | Measures whether mentioned evidence features are actually present in the structured packet. This is the closest fit to evidence-grounded NLG evaluation and attribution-style explanation quality. |
| Hallucination Rate | hallucination / unsupported content | Standard | Widely used in LLM evaluation. Here it captures unsupported features or unsupported claims relative to the packet. |
| Key Evidence Coverage | coverage / evidence recall | Literature-aligned, task-specific wording | The explanation task is tied to active-rule evidence, so coverage is defined over top contributing rules/features rather than over a free-form reference explanation. |
| Prediction-Explanation Agreement | explanation-label consistency / self-consistency | Task-specific | This project needs a direct check that the explanation matches the model's actual predicted class. Generic NLG metrics do not cover that requirement well. |
| Uncertainty Acknowledgment Rate | uncertainty communication / calibration language | Literature-aligned, task-specific operationalization | The system must explicitly acknowledge mixed evidence when rules conflict, so we operationalize uncertainty as language-use conditioned on packet conflict state. |
| Causal Overclaim Rate | factuality / non-causal safety constraint | Task-specific | The task explicitly forbids causal claims, so this metric tracks a project-specific failure mode not usually captured by generic explanation benchmarks. |
| Rule Coverage | evidence coverage / rationale usage | Literature-aligned, task-specific unit | We measure whether the explanation reflects the important active rules, not just the final class. |
| Conflict Acknowledgment Score | contradiction awareness / uncertainty communication | Task-specific | Needed because mixed rule evidence is central to this pipeline and must be surfaced faithfully. |
| Confidence Wording Calibration | calibration / confidence-language alignment | Literature-aligned | The explanation should not sound more certain than the packet evidence supports. |
| Audience Understandability | understandability / comprehensibility | Standard XAI/HCI construct | Matches human-centered XAI evaluation dimensions surveyed in the literature. |
| Audience Technical Fit | audience appropriateness / level-of-detail appropriateness | Task-specific but justified | We need to distinguish “understandable” from “at the right technical level” because outputs are intentionally audience-conditioned. |
| Flesch Reading Ease / Flesch-Kincaid Grade Level | readability | Standard | Used only for layman outputs as a lightweight supporting signal, not as a faithfulness metric. |

## Why Custom Metrics Are Still Needed

The task is not generic generation. It is constrained translation of structured model evidence into audience-specific explanations with explicit non-causal and non-hallucinatory requirements. Because of that:

- standard factuality or readability metrics alone are insufficient
- several metrics must be conditioned on packet structure, such as active-rule conflict
- audience-targeting adds an extra axis not captured by standard faithfulness-only metrics

This is why the v2 metric set combines literature-aligned categories with task-specific operationalizations.

## Judge Model Warning

Judge scores are supportive, not authoritative.

- G-Eval supports LLM-as-judge as a practical reference-free evaluation strategy.
- Later evidence shows judge behavior can vary by model and domain, so judge outputs should not be treated as a ground-truth substitute.

Accordingly, `Audience Understandability` and `Audience Technical Fit` are best interpreted alongside objective packet-grounded metrics rather than in isolation.
