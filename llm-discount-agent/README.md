### 1. AI-Powered Discount Agent

- Building AI agents to identify discount opportunities and calculate discounted prices using locally deployed Qwen3 and pricing tools.

### 2. Reasoning Strategy Evaluation

- Comparing ReAct and Plan-and-Execute based on execution time, model calls, tool calls, errors, price correctness, and potential hallucination rate.
Run the comparison via the CLI:
uv run python src/discount_agent/agent_pipeline.py \
  --reason-type compare \
  "What is the discounted price of a camera?"

### 3. Performance Reporting

- Exporting results to CSV files to analyze agent efficiency, pricing accuracy, and reliability.
