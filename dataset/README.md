# ShopLite synthetic RAG dataset

Eight fictional Markdown policy documents and sixteen evaluation questions for a two-hour educational RAG project. These are not real company policies. No customer data is included.

Index only data/*.md. Keep eval_questions.jsonl separate: it contains expected answers and source labels for evaluating retrieval, grounding, abstention and clarification. Do not index the test questions or reference answers.

All policies have version demo-v1. Each supplied document is fewer than 250 words and can be used as one complete retrieval chunk.

Document topics: returns, refunds, shipping, cancellations, arrival damage, warranty, payments, account support.

Evaluation targets are not measured performance results. Add unseen questions before assessing generalization.
