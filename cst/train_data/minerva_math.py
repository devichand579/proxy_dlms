"""Competition math train load and Minerva few-shot prompt format."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from datasets import Dataset, load_dataset


def _load_minerva_math_train() -> Dataset:
    return load_dataset("qwedsacf/competition_math", split="train")


def list_minerva_math_fewshot_samples() -> List[Dict[str, Any]]:
    """Same four demos as lm-eval ``minerva_math.utils.list_fewshot_samples`` (first_n)."""
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def _minerva_math_format_lm_eval_fixed_fewshot(
    row: Dict[str, Any],
    fewshot: Sequence[Dict[str, Any]],
) -> str:
    """Fixed few-shot demos + train problem (lm-eval Problem/Solution layout)."""
    parts: List[str] = []
    for d in fewshot:
        p = str(d.get("problem", "")).strip()
        s = str(d.get("solution", "")).strip()
        parts.append("Problem:" + "\n" + p + "\n\n" + "Solution: " + s)
    p_test = str(row.get("problem") or row.get("question", "")).strip()
    parts.append("Problem:" + "\n" + p_test + "\n\n" + "Solution:")
    return "\n\n".join(parts)


def _minerva_math_text(row: Dict[str, Any]) -> str:
    p = str(row.get("problem") or row.get("question") or "").strip()
    return "Problem:" + "\n" + p + "\n\n" + "Solution:"
