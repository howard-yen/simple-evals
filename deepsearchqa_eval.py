"""
DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents
https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf

Eval code adopted from: https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code
""" 

import base64
import hashlib
import random
import re
import pandas
import textwrap
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from pydantic import BaseModel
from typing import Literal


QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

DEEPSEARCH_QA_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a JSON dictionary with the following top-level keys: `"explanation"` (a string), `"correctness_details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"excessive_answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
  "correctness_details": {{
    "Belgium": true,
    "France": true,
  }},
  "excessive_answers": [ "Italy" ]
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


class ExtractedResult(BaseModel):
    explanation: str
    correctness_details: dict[str, bool]
    excessive_answers: list[str]
    
class CorrectnessItem(BaseModel):
    key: str
    value: bool

class ExtractedResult(BaseModel):
    explanation: str
    # correctness_details: CorrectnessDetails
    correctness_details: list[CorrectnessItem]
    # correctness_details: Dict[str, bool]
    excessive_answers: list[str]
    strict: Literal[True]


choice_strings = ["yes", "no"]


def _calculate_metric(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    """Calculates precision, recall, and F1."""
    precision_val = 0.0
    if (true_positives + false_positives) > 0:
        precision_val = true_positives / (true_positives + false_positives)

    recall_val = 0.0
    if (true_positives + false_negatives) > 0:
        recall_val = true_positives / (true_positives + false_negatives)

    f1_score_val = 0.0
    if (precision_val + recall_val) > 0:
        f1_score_val = (
            2 * (precision_val * recall_val) / (precision_val + recall_val)
        )

    return {
        'precision': precision_val,
        'recall': recall_val,
        'f1_score': f1_score_val,
    }

    
def calculate_metrics(extracted_result: ExtractedResult) -> dict:
    # Extract correctness details
    details = extracted_result.correctness_details
    expected_correct_answer_list = [item.key for item in details]
    ratings = [item.value for item in details]
    
    num_correct = sum(ratings)
    true_positive = num_correct
    false_negative = len(ratings) - num_correct
    
    has_expected_answers = bool(ratings)

    all_expected_answers_correct = False
    fully_incorrect = 0
    if has_expected_answers:
        all_expected_answers_correct = num_correct == len(ratings)
        if num_correct == 0:
            fully_incorrect = 1

    # Extract excessive answers
    excessive_answers = extracted_result.excessive_answers

    has_excessive_answers = bool(excessive_answers)
    false_positives = 0
    correct_with_excessive_answers = 0
    if has_excessive_answers:
        false_positives = len(excessive_answers)
        if (all_expected_answers_correct or not has_expected_answers):
            correct_with_excessive_answers = 1

    is_all_correct = (
        all_expected_answers_correct or not has_expected_answers
    ) and not has_excessive_answers

    per_item_metric = _calculate_metric(true_positive, false_positives, false_negative)

    return {
        "all_correct": is_all_correct,
        "correct_with_excessive_answers": correct_with_excessive_answers,
        "fully_incorrect": fully_incorrect,
        **per_item_metric,
    }


# Note: option to use the local path
LOCAL_PATH = "simple-evals/data/DSQA-full.csv"

class DeepSearchQAEval(Eval):
    def __init__(self, grader_model: SamplerBase, num_examples: int | None = None, n_repeats: int = 1, n_threads: int = 1):
        df = pandas.read_csv(LOCAL_PATH)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.grader_model = grader_model
        self.n_threads = n_threads

    def grade_sample(self, question: str, prompt_type: str, correct_answer: str, response: str) -> dict:
        grader_prompt = DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
            prompt=question,
            prompt_type=prompt_type,
            answer=correct_answer,
            response=response,
        )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        extracted_result = self.grader_model.parse(prompt_messages, ExtractedResult)
        if extracted_result is None:
            return None

        metrics = calculate_metrics(extracted_result)
        return metrics

    def __call__(self, sampler: SamplerBase, checkpoint_file: str | None = None, checkpoint_interval: int = 10) -> EvalResult:
        def fn(row: dict):
            problem = row.get("problem", "")
            answer = row.get("answer", "")
            answer_type = row.get("answer_type", "")
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(Question=problem), role="user")
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            grade_result = self.grade_sample(problem, answer_type, answer, response_text)

            # Metrics based on grading response
            score = grade_result["all_correct"]

            model_response = [dict(content=response_text, role="assistant", type="text")]
            if "extra_convo" in sampler_response.response_metadata:
                model_response = sampler_response.response_metadata.pop("extra_convo") + model_response

            # Create HTML for each sample result
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=model_response,
                extracted_answer=response_text,
                score=score,
                correct_answer=answer,
            )
            convo = actual_queried_prompt_messages + model_response
            grade_result["response_text"] = response_text
            grade_result["response_metadata"] = sampler_response.response_metadata

            grade_result["question"] = problem
            grade_result["answer"] = answer

            return SingleEvalResult(html=html, score=score, convo=convo, metrics={
                "fully_correct": grade_result["all_correct"],
                "correct_with_excessive_answers": grade_result["correct_with_excessive_answers"],
                "fully_incorrect": grade_result["fully_incorrect"],
                "precision": grade_result["precision"],
                "recall": grade_result["recall"],
                "f1_score": grade_result["f1_score"],
            }, example_level_metadata=grade_result)

        # Run evaluation and collect results
        results = common.map_with_progress(fn, self.examples, num_threads=self.n_threads, checkpoint_file=checkpoint_file, checkpoint_interval=checkpoint_interval)

        # Aggregate metrics
        aggregate_metrics = {
            "fully_correct": sum(result.metrics["fully_correct"] for result in results) / len(results),
            "correct_with_excessive_answers": sum(result.metrics["correct_with_excessive_answers"] for result in results) / len(results),
            "fully_incorrect": sum(result.metrics["fully_incorrect"] for result in results) / len(results),
            "precision": sum(result.metrics["precision"] for result in results) / len(results),
            "recall": sum(result.metrics["recall"] for result in results) / len(results),
            "f1_score": sum(result.metrics["f1_score"] for result in results) / len(results),
        }
        print("AGGREGATE METRICS") 
        print(aggregate_metrics) 
        print("##################")

        output_d = {
            "fully_correct": aggregate_metrics["fully_correct"],
            "correct_with_excessive_answers": aggregate_metrics["correct_with_excessive_answers"],
            "fully_incorrect": aggregate_metrics["fully_incorrect"],
            "precision": aggregate_metrics["precision"],
            "recall": aggregate_metrics["recall"],
            "f1_score": aggregate_metrics["f1_score"],
        }
        
        print(f"Fully Correct: {output_d['fully_correct']:.3f}\nCorrect With Excessive Answers: {output_d['correct_with_excessive_answers']:.3f}\nFully Incorrect: {output_d['fully_incorrect']:.3f}\nPrecision: {output_d['precision']:.3f}\nRecall: {output_d['recall']:.3f}\nF1 Score: {output_d['f1_score']:.3f}")
        
        return common.aggregate_results(results)
