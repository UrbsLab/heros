# HEROS-LLM v2 Metric Memo

This memo documents the revised metric stack after shifting the project toward literature-based metric names wherever possible. The goal is to use standard evaluation constructs as the default and reserve task-specific metrics only for failure modes that are not well covered by prior work.

## Evaluation Framing

Following Doshi-Velez and Kim, the evaluation stack is intentionally split across two levels:

- `Functionally grounded` evaluation: objective metrics computed directly from the structured rule packet, such as `Evidence Precision`, `Evidence Recall`, `Evidence F1`, `Hallucination Rate`, `Comprehensiveness`, and `Sufficiency`.
- `Human-grounded` evaluation: audience-facing quality judgments such as `Audience Understandability` and `Audience Technical Fit`, implemented here with an LLM judge as a scalable proxy rather than a replacement for human evaluation.

This separation matters because the project is not trying to evaluate free-form explanation quality in the abstract. It is evaluating whether audience-specific natural-language explanations stay faithful to structured model evidence while remaining understandable at the intended technical level.

## Anchor References

- Doshi-Velez and Kim, *Towards A Rigorous Science of Interpretable Machine Learning* (2017/2018): [arXiv](https://arxiv.org/abs/1702.08608)
- Kim et al., *Human-centered evaluation of explainable AI applications: a systematic review* (2024): [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1456486/full)
- DeYoung et al., *ERASER: A Benchmark to Evaluate Rationalized NLP Models* (ACL 2020): [ACL Anthology](https://aclanthology.org/2020.acl-main.408/)
- Atanasova et al., *Faithfulness Tests for Natural Language Explanations* (ACL 2023): [ACL Anthology](https://aclanthology.org/2023.acl-short.25/)
- Parcalabescu and Frank, *On Measuring Faithfulness or Self-consistency of Natural Language Explanations* (ACL 2024): [ACL Anthology](https://aclanthology.org/2024.acl-long.329/)
- Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* (EMNLP 2023): [ACL Anthology](https://aclanthology.org/2023.emnlp-main.153/)
- Huang et al., *An Empirical Study of LLM-as-a-Judge for LLM Evaluation* (Findings ACL 2025): [ACL Anthology](https://aclanthology.org/2025.findings-acl.306/)
- Bang et al., *HalluLens: LLM Hallucination Benchmark* (ACL 2025): [ACL Anthology](https://aclanthology.org/2025.acl-long.1176/)
- Lertvittayakumjorn et al., *Diagnostics-Guided Explanation Generation* (AAAI 2022): [AAAI PDF](https://cdn.aaai.org/ojs/21287/21287-13-25300-1-2-20220628.pdf)
- Xing et al., *Towards Faithful Model Explanation in Natural Language with Evidence Attribution* (NAACL 2025): [ACL Anthology](https://aclanthology.org/2025.naacl-long.282/)

## Primary Literature-Based Metrics

| Current name | Prior construct | Status | Why we use it here |
| --- | --- | --- | --- |
| Evidence Precision | rationale precision / evidence attribution precision | Literature-based name, task-specific operationalization | Measures how many explicitly mentioned features belong to the rule-derived evidence set. This is the closest fit to evidence precision in rationale and attribution evaluation. |
| Evidence Recall | rationale recall / evidence attribution recall | Literature-based name, task-specific operationalization | Measures how much of the rule-derived evidence set is recovered by the explanation. |
| Evidence F1 | precision/recall harmonic mean | Standard derived metric | Summarizes the evidence precision/recall tradeoff in one value. |
| Hallucination Rate | hallucination / unsupported content | Standard | Widely used in LLM evaluation. Here it captures unsupported features or unsupported claims relative to the packet. |
| Comprehensiveness | rationale faithfulness under removal | Literature-based name, scaffolded only for now | Included because it is a standard perturbation-based faithfulness metric. It is not numerically computed yet because the current pipeline does not rerun the classifier under feature removal. |
| Sufficiency | rationale faithfulness under keep-only evidence | Literature-based name, scaffolded only for now | Included because it is a standard perturbation-based faithfulness metric. It is not numerically computed yet because the current pipeline does not rerun the classifier under keep-only perturbations. |
| Flesch Reading Ease / Flesch-Kincaid Grade Level | readability | Standard | Used only for layman outputs as lightweight literature-based readability signals. |

## Secondary Task-Specific Metrics

These remain in the codebase because the project has constraints that are not fully covered by standard explanation-generation metrics.

| Metric | Prior construct | Status | Why we still keep it |
| --- | --- | --- | --- |
| Prediction-Explanation Agreement | explanation-label consistency / self-consistency | Task-specific | This project needs a direct check that the explanation matches the model's actual predicted class. Generic NLG metrics do not cover that requirement well. |
| Uncertainty Acknowledgment Rate | uncertainty communication / calibration language | Literature-aligned, task-specific operationalization | The system must explicitly acknowledge mixed evidence when rules conflict, so we operationalize uncertainty as language-use conditioned on packet conflict state. |
| Causal Overclaim Rate | factuality / non-causal safety constraint | Task-specific | The task explicitly forbids causal claims, so this metric tracks a project-specific failure mode not usually captured by generic explanation benchmarks. |
| Rule Coverage | evidence coverage / rationale usage | Literature-aligned, task-specific unit | We measure whether the explanation reflects the important active rules, not just the final class. |
| Conflict Acknowledgment Score | contradiction awareness / uncertainty communication | Task-specific | Needed because mixed rule evidence is central to this pipeline and must be surfaced faithfully. |
| Confidence Wording Calibration | calibration / confidence-language alignment | Literature-aligned | The explanation should not sound more certain than the packet evidence supports. |
| Audience Understandability | understandability / comprehensibility | Standard XAI/HCI construct | Matches human-centered XAI evaluation dimensions surveyed in the literature. |
| Audience Technical Fit | audience appropriateness / level-of-detail appropriateness | Task-specific but justified | We need to distinguish “understandable” from “at the right technical level” because outputs are intentionally audience-conditioned. |

## Why Some Custom Metrics Are Still Needed

The task is not generic generation. It is constrained translation of structured model evidence into audience-specific explanations with explicit non-causal and non-hallucinatory requirements. Because of that:

- standard rationale and factuality metrics alone are insufficient
- several metrics must be conditioned on packet structure, such as active-rule conflict
- audience-targeting adds an extra axis not captured by faithfulness-only metrics

This is why the revised metric set uses literature-based names for the primary evidence metrics while retaining a smaller set of task-specific safety and audience metrics.

## Judge Model Warning

Judge scores are supportive, not authoritative.

- G-Eval supports LLM-as-judge as a practical reference-free evaluation strategy.
- Later evidence shows judge behavior can vary by model and domain, so judge outputs should not be treated as a ground-truth substitute.

Accordingly, `Audience Understandability` and `Audience Technical Fit` are best interpreted alongside objective packet-grounded metrics rather than in isolation.
