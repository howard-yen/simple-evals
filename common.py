import io
import os
import pickle
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import jinja2
import numpy as np
import requests
from tqdm import tqdm

from .types import EvalResult, Message, SamplerBase, SingleEvalResult

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
]


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{% if next_message is mapping %}
{{ message_to_html(next_message) | safe }}
{% else %}
{% for message in next_message %}
{{ message_to_html(message) | safe }}
{% endfor %}
{% endif %}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
{% if extra_info|default(false) %}
<h3>Additional Information</h3>
{% for key, value in extra_info.items() %}
<p>{{ key }}: {{ value }}</p>
{% endfor %}
{% endif %}
"""



def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def check_equality(sampler: SamplerBase, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    sampler_response = sampler([dict(content=prompt, role="user")])
    response_text = sampler_response.response_text
    return response_text.lower().strip() == "yes"


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
    checkpoint_file: str | None = None,
    checkpoint_interval: int = 10,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    
    Args:
        f: Function to apply to each element
        xs: List of inputs to process
        num_threads: Number of threads to use
        pbar: Whether to show progress bar
        checkpoint_file: Optional file path to save/load checkpoints
        checkpoint_interval: Save checkpoint every N completed items
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x
    
    # Load existing results from checkpoint if it exists
    completed_results = {}
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as fin:
                completed_results = pickle.load(fin)
                if pbar:
                    print(f"Loaded checkpoint with {len(completed_results)} completed items")
        except Exception as e:
            if pbar:
                print(f"Warning: Could not load checkpoint file: {e}")
            completed_results = {}
    
    # Filter out already completed items
    remaining_items = [(i, x) for i, x in enumerate(xs) if i not in completed_results]
    
    if not remaining_items:
        # All items already completed
        return [completed_results[i] for i in range(len(xs))]
    
    # Set up checkpoint saving
    checkpoint_lock = threading.Lock() if checkpoint_file else None
    
    def save_checkpoint():
        # only call this if the lock is already acquired
        try:
            # Write to temporary file first, then rename for atomicity
            temp_file = checkpoint_file + '.tmp'
            with open(temp_file, 'wb') as fout:
                pickle.dump(completed_results, fout)
            os.rename(temp_file, checkpoint_file)
        except Exception as e:
            if pbar:
                print(f"Warning: Could not save checkpoint: {e}")
    
    def wrapped_f(item_tuple):
        i, x = item_tuple
        try:
            result = f(x)
            
            # Save result and update checkpoint if needed
            if checkpoint_file:
                with checkpoint_lock:
                    completed_results[i] = result
                    
                    # Save checkpoint periodically
                    if len(completed_results) % checkpoint_interval == 0:
                        save_checkpoint()
            
            return i, result
        except Exception as e:
            # Re-raise with index information for debugging
            raise Exception(f"Error processing item {i}: {str(e)}") from e

    if os.getenv("debug"):
        # Sequential execution for debugging
        results_with_indices = []
        for item in pbar_fn(remaining_items, total=len(remaining_items)):
            results_with_indices.append(wrapped_f(item))
    else:
        # Parallel execution
        with ThreadPool(min(num_threads, len(remaining_items))) as pool:
            results_with_indices = list(pbar_fn(
                pool.imap(wrapped_f, remaining_items), 
                total=len(remaining_items)
            ))
    
    # Update completed results with new results
    for i, result in results_with_indices:
        completed_results[i] = result
    
    # Save final checkpoint
    if checkpoint_file:
        save_checkpoint()
    
    # Return results in original order
    return [completed_results[i] for i in range(len(xs))]


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Generate HTML snippet (inside a <div>) for a message.
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Create a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )


def make_report_from_example_htmls(htmls: list[str]):
    """
    Create a standalone HTML report from a list of example htmls
    """
    return jinja_env.from_string(_report_template).render(
        score=None, metrics={}, htmls=htmls
    )


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def url_to_fileobj(url: str, binary=False) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content) if binary else io.StringIO(response.text)


def has_only_user_assistant_messages(messages: list[Message]) -> bool:
    """
    Check if the messages only contain user and assistant messages.
    """
    return all(m["role"] in ("user", "assistant") for m in messages)


def get_usage_dict(response_usage) -> dict[str, int | None]:
    if response_usage is None:
        return {
            "input_tokens": None,
            "input_cached_tokens": None,
            "output_tokens": None,
            "output_reasoning_tokens": None,
            "total_tokens": None,
        }

    try:
        return {
            "input_tokens": response_usage.input_tokens,
            "input_cached_tokens": (response_usage.input_tokens_details.cached_tokens
            if hasattr(response_usage.input_tokens_details, "cached_tokens")
            else response_usage.input_tokens_details["cached_tokens"])
            if hasattr(response_usage, "input_tokens_details") and response_usage.input_tokens_details is not None
            else None,
            "output_tokens": response_usage.output_tokens,
            "output_reasoning_tokens": (response_usage.output_tokens_details.reasoning_tokens
            if hasattr(response_usage.output_tokens_details, "reasoning_tokens")
            else response_usage.output_tokens_details["reasoning_tokens"])
            if hasattr(response_usage, "output_tokens_details") and response_usage.output_tokens_details is not None
            else None,
            "total_tokens": response_usage.total_tokens,
        }
    except AttributeError:
        return {
            "input_tokens": response_usage.prompt_tokens,
            "input_cached_tokens": (response_usage.prompt_tokens_details.cached_tokens
            if hasattr(response_usage.prompt_tokens_details, "cached_tokens")
            else response_usage.prompt_tokens_details["cached_tokens"])
            if hasattr(response_usage, "prompt_tokens_details") and response_usage.prompt_tokens_details is not None
            else None,
            "output_tokens": response_usage.completion_tokens,
            "output_reasoning_tokens": (response_usage.completion_tokens_details.reasoning_tokens
            if hasattr(response_usage.completion_tokens_details, "reasoning_tokens")
            else response_usage.completion_tokens_details["reasoning_tokens"])
            if hasattr(response_usage, "completion_tokens_details") and response_usage.completion_tokens_details is not None
            else None,
            "total_tokens": response_usage.total_tokens,
        }

