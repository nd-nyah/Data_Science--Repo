import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable


load_dotenv()

MAX_ITER = 10
MODEL = "qwen3:1.7b"
CSV_PATH = "agent_comparison.csv"

# Ground truth for this educational pricing example.
CATALOG = {
    "camera": 299.00,
    "pen": 20.00,
    "mouse": 50.00,
}
DISCOUNTS = {
    "gold": 20,
    "silver": 15,
}


# ============================================================
# Tools
# ============================================================

@tool
def get_product_price(product: str) -> float:
    """Get the base price of a product from the catalog."""
    print(f">>> get_product_price(product={product!r})")
    return CATALOG.get(product.strip().lower(), 0.0)


@tool
def apply_discount(price: float, discount_type: str) -> float:
    """Apply a gold (20%) or silver (15%) discount to a price."""
    print(
        f">>> apply_discount(price={price}, "
        f"discount_type={discount_type!r})"
    )
    percentage = DISCOUNTS.get(discount_type.strip().lower(), 0)
    return round(float(price) * (1 - percentage / 100), 2)


# ============================================================
# Metrics
# ============================================================

@dataclass
class PerformanceTracker:
    reasoning_type: str
    start_time: float = field(default_factory=time.perf_counter)
    model_calls: int = 0
    tool_calls: int = 0
    errors: int = 0

    def record_model_call(self, response: Any) -> None:
        self.model_calls += 1

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_error(self) -> None:
        self.errors += 1

    def report(self) -> dict:
        return {
            "reasoning_type": self.reasoning_type,
            "execution_time_seconds": round(
                time.perf_counter() - self.start_time, 3
            ),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
        }



def invoke_model(llm, messages, metrics):
    metrics.record_model_call(None)

    try:
        response = llm.invoke(messages)
        print("Model response received successfully.")
        print("Tool calls requested:", response.tool_calls)
        return response
    except Exception as exc:
        metrics.record_error()
        print(f"\nMODEL ERROR: {type(exc).__name__}: {exc}")
        raise



def invoke_tool_call(call, tools_dict, metrics: PerformanceTracker):
    """Run one tool call and return its result as text."""
    metrics.record_tool_call()

    name = call.get("name", "")
    args = call.get("args", {})

    print(f"\nTool: {name}")
    print(f"Arguments: {args}")

    selected_tool = tools_dict.get(name)
    if selected_tool is None:
        metrics.record_error()
        return f"Tool error: unknown tool {name!r}"

    # Validate the input price before applying a discount.
    if name == "apply_discount":
        try:
            price = float(args.get("price", -1))
        except (TypeError, ValueError):
            metrics.record_error()
            return (
                "Tool error: invalid or missing price. "
                "Call get_product_price first and use its returned price."
            )

        valid_prices = CATALOG.values()

        if not any(abs(price - p) < 0.01 for p in valid_prices):
            metrics.record_error()
            return (
                f"Tool error: price {price} does not match a catalog price. "
                "Call get_product_price first and use its returned value."
            )

    try:
        result = selected_tool.invoke(args)
        print(f"Observation: {result}")
        return str(result)

    except Exception as exc:
        metrics.record_error()
        message = f"Tool execution failed: {exc}"
        print(message)
        return message


def execute_tool_calls(response, messages, tools_dict, metrics):
    """Execute all tool calls from a model response and append observations."""
    for call in response.tool_calls:
        result = invoke_tool_call(call, tools_dict, metrics)
        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=call["id"],
            )
        )


# ============================================================
# Reasoning strategy: ReAct
# ============================================================

def run_react(question, llm_tools, tools_dict, metrics):
    """Iteratively call tools, observe results, and continue."""
    print("\n[Reasoning strategy: ReAct]")

    messages = [
        SystemMessage(
            content=(
                "You are a product pricing assistant. Use tools for pricing. "
                "For a discounted price, first retrieve the base price, then "
                "apply the requested discount. If no discount is specified, "
                "use gold. Use tool observations as the source of truth. "
                "Do not invent prices. A catalog lookup returning 0 means "
                "the product was not found."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITER):
        print(f"\n--- ReAct iteration {iteration + 1} ---")
        try:
            response = invoke_model(llm_tools, messages, metrics)
        except Exception as exc:
            return f"Model invocation failed: {exc}"

        messages.append(response)

        if not response.tool_calls:
            answer = str(response.content)
            print("\nFinal answer:", answer)
            return answer

        execute_tool_calls(response, messages, tools_dict, metrics)

    metrics.record_error()
    return "Agent stopped: maximum iteration limit reached."


# ============================================================
# Reasoning strategy: Plan-and-Execute
# ============================================================

def run_plan_execute(question, llm, llm_tools, tools_dict, metrics):
    """Generate a plan, execute its steps, then synthesize an answer."""
    print("\n[Reasoning strategy: Plan-and-Execute]")

    planner_prompt = f"""
You plan tasks for a product pricing system.

User question:
{question}

Available tools:
- get_product_price(product): retrieves a product's base price.
- apply_discount(price, discount_type): applies gold or silver discount.

Create a short ordered plan. For discounted prices, retrieve the base price
before applying a discount. If the user does not specify a discount, use gold.
Return only a JSON array of short step descriptions, for example:
["Get the laptop base price",
 "Apply the gold discount using the returned price",
 "Report the discounted price"]
"""

    try:
        plan_response = invoke_model(llm, planner_prompt, metrics)
    except Exception as exc:
        metrics.record_error()
        return f"Planning model invocation failed: {exc}"

    plan_text = str(plan_response.content).strip()
    plan_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", plan_text).strip()

    try:
        plan = json.loads(plan_text)
        if not isinstance(plan, list) or not plan:
            raise ValueError("Plan must be a non-empty JSON list")
        plan = [str(step) for step in plan[:MAX_ITER]]
    except (json.JSONDecodeError, ValueError):
        print("Planner output was not valid JSON; using fallback plan.")
        plan = [
            "Identify the product and requested discount",
            "Retrieve the product base price",
            "Apply the requested discount if applicable",
            "Report the result",
        ]

    print("\nGenerated plan:")
    for index, step in enumerate(plan, start=1):
        print(f"  {index}. {step}")

    observations = []
    total_execution_model_calls = 0

    for index, step in enumerate(plan, start=1):
        if total_execution_model_calls >= MAX_ITER:
            print("Execution limit reached.")
            metrics.record_error()
            break

        print(f"\n--- Plan step {index}: {step} ---")
        messages = [
            SystemMessage(
                content=(
                    "Execute the current step of a product-pricing plan. "
                    "Use the provided tools when needed. Use prior observations "
                    "and do not invent product prices. If the step is only "
                    "summarization, respond without tools."
                )
            ),
            HumanMessage(
                content=(
                    f"Original question: {question}\n"
                    f"Current step: {step}\n"
                    f"Prior observations: "
                    f"{json.dumps(observations, ensure_ascii=False)}\n"
                    "Perform this step."
                )
            ),
        ]

        for _ in range(MAX_ITER - total_execution_model_calls):
                    try:
                        response = invoke_model(llm_tools, messages, metrics)
                    except Exception as exc:
                        observations.append({
                            "step": step,
                            "error": str(exc),
                        })
                        break

                    total_execution_model_calls += 1
                    messages.append(response)

                    if not response.tool_calls:
                        observations.append({
                            "step": step,
                            "result": str(response.content),
                        })
                        print("Step result:", response.content)
                        break

                    # Preserve each tool observation in the conversation and
                    # in the structured record used by later steps.
                    step_results = []
                    for call in response.tool_calls:
                        result = invoke_tool_call(call, tools_dict, metrics)
                        messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=call["id"],
                            )
                        )
                        step_results.append({
                            "tool": call.get("name"),
                            "args": call.get("args", {}),
                            "result": result,
                        })

                    observations.append({
                        "step": step,
                        "tool_results": step_results,
                    })

    print("\n--- Final synthesis ---")
    final_prompt = [
        SystemMessage(
            content=(
                "Answer the user's original question using the execution "
                "observations. Do not invent prices. If the product is missing "
                "or required evidence is absent, explain that limitation."
            )
        ),
        HumanMessage(
            content=(
                f"Original question: {question}\n"
                f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
                f"Execution observations: "
                f"{json.dumps(observations, ensure_ascii=False)}"
            )
        ),
    ]

    try:
        final_response = invoke_model(llm, final_prompt, metrics)
        answer = str(final_response.content)
    except Exception as exc:
        metrics.record_error()
        answer = f"Final synthesis failed: {exc}"

    print("\nFinal answer:", answer)
    return answer


# ============================================================
# Hallucination / factual evaluation for pricing questions
# ============================================================

def expected_price_for_question(question: str):
    """
    Derive the expected price from the trusted local catalog.
    Returns None if the question does not match the supported pricing task.
    """
    text = question.lower()

    product = next(
        (name for name in CATALOG if re.search(rf"\b{re.escape(name)}\b", text)),
        None,
    )
    if product is None:
        return None

    # Only apply a discount when the question asks for a discounted price
    # or explicitly mentions a discount type.
    asks_discount = any(
        phrase in text
        for phrase in ("discount", "discounted", "after discount", "reduced price")
    )
    if not asks_discount:
        return CATALOG[product]

    discount_type = "silver" if "silver" in text else "gold"
    percentage = DISCOUNTS[discount_type]
    return round(CATALOG[product] * (1 - percentage / 100), 2)


def evaluate_hallucination(answer: str, question: str) -> dict:
    """
    Evaluate numeric claims in Qwen's answer against trusted catalog data.

    No manual True/False labels are assigned. Accuracy and hallucination
    rates are calculated from extracted claims and reference values.

    This is a lightweight numeric evaluator, not a general semantic
    hallucination detector.
    """
    text = str(answer)
    question_lower = question.lower()

    product = next(
        (
            name for name in CATALOG
            if re.search(rf"\b{re.escape(name)}\b", question_lower)
        ),
        None,
    )

    if product is None:
        return {
            "product": None,
            "verifiable_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "factual_accuracy_rate": None,
            "hallucination_rate": None,
            "evaluation_note": "Product not in the supported catalog.",
        }

    base_price = CATALOG[product]

    asks_discount = any(
        phrase in question_lower
        for phrase in (
            "discount",
            "discounted",
            "after discount",
            "reduced price",
        )
    )

    discount_type = "silver" if "silver" in question_lower else "gold"
    discount_percent = DISCOUNTS[discount_type]
    discounted_price = round(
        base_price * (1 - discount_percent / 100), 2
    )

    # Trusted reference values are derived from the local catalog/rules.
    valid_prices = {round(base_price, 2)}
    valid_percentages = set()
    valid_discount_amounts = set()

    if asks_discount:
        discount_amount = round(base_price - discounted_price, 2)
        
        valid_prices.add(discounted_price)
        valid_percentages.add(float(discount_percent))

    claims = []

    # Extract percentage claims such as "20%" or "discount is 15 percent".
    percentage_pattern = (
        r"(?<!\w)(\d+(?:\.\d+)?)\s*(?:%|percent\b)"
    )

    for match in re.finditer(percentage_pattern, text, re.I):
        claims.append({
            "type": "percentage",
            "value": float(match.group(1)),
        })

    # Extract currency amounts such as "$299.00", "£239.20", or "299.00".
    currency_pattern = (
        r"[$£€]\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)"
    )

    for match in re.finditer(currency_pattern, text):
        value = float(match.group(1).replace(",", ""))
        claims.append({"type": "price", "value": value})

    # Also recognize uncurrency-marked amounts near price-related words.
    contextual_price_pattern = (
        r"\b(?:price|cost|total|pay|pays|paying|"
        r"costs|base price|final price)\b"
        r"[^.\n]{0,35}?"
        r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d{1,2})?)"
    )

    for match in re.finditer(contextual_price_pattern, text, re.I):
        value = float(match.group(1).replace(",", ""))
        claims.append({"type": "price", "value": value})

    # Remove duplicate claims extracted by overlapping patterns.
    unique_claims = []
    seen = set()

    for claim in claims:
        key = (claim["type"], round(claim["value"], 2))
        if key not in seen:
            seen.add(key)
            unique_claims.append(claim)

    supported = 0
    unsupported = 0

    for claim in unique_claims:
        value = claim["value"]

        
        if claim["type"] == "price":
            is_supported = any(
                abs(value - expected) < 0.01
                for expected in valid_prices
            )

        elif claim["type"] == "percentage":
            is_supported = any(
                abs(value - expected) < 0.01
                for expected in valid_percentages
            )

        else:
            is_supported = False


        if is_supported:
            supported += 1
        else:
            unsupported += 1

    total = supported + unsupported

    accuracy_rate = (
        round(supported / total * 100, 2) if total else None
    )
    hallucination_rate = (
        round(unsupported / total * 100, 2) if total else None
    )

    expected = (
        discounted_price if asks_discount else base_price
    )

    # This checks whether the expected answer price appears in the output.
    answer_has_expected_price = any(
        claim["type"] == "price"
        and abs(claim["value"] - expected) < 0.01
        for claim in unique_claims
    )

    return {
        "product": product,
        "expected_price": expected,
        "expected_discount_percent": (
            discount_percent if asks_discount else None
        ),
        "verifiable_claims": total,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "factual_accuracy_rate": accuracy_rate,
        "hallucination_rate": hallucination_rate,
        "expected_price_present": answer_has_expected_price,
        "evaluation_note": (
            "Rates are based on extracted numeric claims matched against "
            "the trusted catalog and discount rules. They do not measure "
            "all semantic or non-numeric hallucinations."
        ),
        "price_correct": answer_has_expected_price,
        "potential_hallucination": (
            round(unsupported / total * 100, 2) if total else None
        ),
    }


# ============================================================
# Agent runner and comparison
# ============================================================

@traceable(name="compare_agent_reasoning")
def run_agent(question: str, reason_type: str) -> dict:
    tools = [get_product_price, apply_discount]
    tools_dict = {item.name: item for item in tools}

    llm = init_chat_model(
        f"ollama:{MODEL}",
        temperature=0,
        seed =42,
    )
    llm_tools = llm.bind_tools(tools)
    metrics = PerformanceTracker(reasoning_type=reason_type)

    print("\n" + "=" * 65)
    print(f"Question: {question}")
    print(f"Model: {MODEL}")
    print(f"Reasoning: {reason_type}")
    print("=" * 65)

    if reason_type == "react":
        answer = run_react(question, llm_tools, tools_dict, metrics)
    elif reason_type == "plan-and-execute":
        answer = run_plan_execute(question, llm, llm_tools, tools_dict, metrics)
    else:
        raise ValueError(f"Unsupported reasoning type: {reason_type}")

    performance = metrics.report()
    hallucination = evaluate_hallucination(answer, question)

    result = {
        "reasoning_type": reason_type,
        "question": question,
        "answer": answer,
        "performance": performance,
        "hallucination": hallucination,
    }

    print("\n--- Performance ---")
    for key, value in performance.items():
        print(f"{key}: {value}")

    print("\n--- Hallucination / factual checks ---")
    for key, value in hallucination.items():
        print(f"{key}: {value}")

    return result

def print_comparison(results: list[dict]) -> None:
    print("\n\n" + "=" * 105)
    print("REASONING STRATEGY COMPARISON")
    print("=" * 105)

    columns = [
        ("Strategy", 20),
        ("Time(s)", 10),
        ("Model calls", 13),
        ("Tool calls", 12),
        ("Errors", 8),
        ("Price correct", 14),
        ("Potential hallucination", 24),
        
    ]
    print("".join(f"{name:<{width}}" for name, width in columns))
    print("-" * 105)

    for result in results:
        p = result["performance"]
        h = result["hallucination"]
        values = [
            result["reasoning_type"],
            p["execution_time_seconds"],
            p["model_calls"],
            p["tool_calls"],
            p["errors"],
            h["price_correct"],
            h["potential_hallucination"],
            
        ]
        print("".join(
            f"{str(value):<{columns[index][1]}}"
            for index, value in enumerate(values)
        ))

    print("=" * 105)


def save_csv(results: list[dict], filename: str) -> None:
    
    fields = [
        "reasoning_type",
        "question",
        "answer",
        "execution_time_seconds",
        "model_calls",
        "tool_calls",
        "errors",
        "product",
        "expected_price",
        "expected_discount_percent",
        "verifiable_claims",
        "supported_claims",
        "unsupported_claims",
        "factual_accuracy_rate",
        "hallucination_rate",
        "expected_price_present",
        "evaluation_note",
        "price_correct",
        "potential_hallucination",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            p = result["performance"]
            h = result["hallucination"]
            writer.writerow({
                "reasoning_type": result["reasoning_type"],
                "question": result["question"],
                "answer": result["answer"],
                **p,
                **h,
            })

    print(f"\nSaved results to: {filename}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare ReAct and Plan-and-Execute pricing agents."
    )
    parser.add_argument(
        "--reason-type",
        choices=["react", "plan-and-execute", "compare"],
        default="react",
        help="Reasoning strategy; use compare to run both.",
    )
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help=f"CSV output path (default: {CSV_PATH})",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask the agent.",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = "What is the discounted price of a laptop?"

    if args.reason_type == "compare":
        strategies = ["react", "plan-and-execute"]
    else:
        strategies = [args.reason_type]

    results = []
    for strategy in strategies:
        results.append(run_agent(question, strategy))

    if len(results) > 1:
        print_comparison(results)

    save_csv(results, args.csv)


if __name__ == "__main__":
    main()