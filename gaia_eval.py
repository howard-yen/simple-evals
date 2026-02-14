import base64
import hashlib
import random
import re
import pandas
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
import json

QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# from: https://github.com/Alibaba-NLP/DeepResearch/blob/76bc781990d687e921b10043f79a7f9c2d81b5b2/evaluation/prompt.py#L98
GRADER_TEMPLATE = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
""".strip()

CHOICE_STRINGS = ["Correct", "Incorrect"]

# Note: option to use the local path
LOCAL_PATH = "simple-evals/data/gaia_standardized.jsonl"
# LOCAL_PATH = None

class GaiaEval(Eval):
    def __init__(self, grader_model: SamplerBase, num_examples: int | None = None, n_repeats: int = 1, n_threads: int = 1):

        df = pandas.read_json(LOCAL_PATH if LOCAL_PATH else "", lines=True)

        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples and num_examples < len(examples):
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.grader_model = grader_model
        self.n_threads = n_threads

    def grade_sample(self, question: str, correct_answer: str, response: str) -> dict:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = self.grader_model(prompt_messages)
        grading_response = sampler_response.response_text

        # Parse correctness
        correct_match = re.search(r"(Correct|Incorrect)", grading_response)
        correctness = correct_match.group(1) if correct_match else "Incorrect"

        return {
            "correctness": correctness
        }

    def __call__(self, sampler: SamplerBase, checkpoint_file: str | None = None, checkpoint_interval: int = 10) -> EvalResult:
        def fn(row: dict):
            problem = row.get("task_question", "")
            answer = row.get("ground_truth", "")
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(Question=problem), role="user")
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            grade_result = self.grade_sample(problem, answer, response_text)

            # Metrics based on grading response
            is_correct = grade_result["correctness"] == "Correct"

            score = is_correct

            model_response = [dict(content=response_text, role="assistant", type="text")]
            if "extra_convo" in sampler_response.response_metadata:
                model_response = sampler_response.response_metadata.pop("extra_convo") + model_response

            # Create HTML for each sample result
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=model_response,
                score=score,
                correct_answer=answer,
                extracted_answer=response_text,
            )
            convo = actual_queried_prompt_messages + model_response
            grade_result["response_text"] = response_text
            grade_result["response_metadata"] = sampler_response.response_metadata

            grade_result["question"] = problem
            grade_result["answer"] = answer

            return SingleEvalResult(html=html, score=score, convo=convo, metrics={
                "is_correct": is_correct,
            }, example_level_metadata=grade_result)

        # Run evaluation and collect results
        results = common.map_with_progress(fn, self.examples, num_threads=self.n_threads, checkpoint_file=checkpoint_file, checkpoint_interval=checkpoint_interval)

        # Aggregate metrics
        aggregate_metrics = {
            "is_correct": sum(result.metrics["is_correct"] for result in results) / len(results),
        }
        print("AGGREGATE METRICS")
        print(aggregate_metrics)
        print("##################")

        output_d = {
            "accuracy": aggregate_metrics["is_correct"],
        }

        print(f"Accuracy: {output_d['accuracy']:.3f}")

        return common.aggregate_results(results)