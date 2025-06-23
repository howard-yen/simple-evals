"""
HLE (Human-Level Evaluation) benchmark
Data loading from CAIS HLE dataset
"""

import math
import numpy as np
import random
import re
from datasets import load_dataset
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from typing import Literal

SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.""".strip()


def format_message(question):
    answer_type = question['answer_type']
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == 'exact_match' else SYSTEM_MC
    question_text = question['question']
    
    message = f"{system_prompt}\n\n{question_text}"
    return message


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p='2', beta=100): 
    # beta is target bin size
    # HY: if beta >= len(confidence), then the error will be 0
    beta = min(beta, len(confidence)-1)
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


class HLEEval(Eval):
    def __init__(self, grader_model: SamplerBase, num_examples: int | None = None, n_repeats: int = 1, n_threads: int = 1, subset_name: Literal["text", "all"] | None = None):
        # Data loading (keeping the original HLE data loading logic)
        dataset = load_dataset("cais/hle", split="test")
        if subset_name == "text":
            dataset = dataset.filter(lambda x: not x["image"])

        dataset = dataset.to_dict()
        examples = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.n_threads = n_threads
        self.grader_model = grader_model

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

        # Parse extracted_final_answer
        extracted_final_answer_match = re.search(r"extracted_final_answer: (.+)", grading_response)
        extracted_final_answer = extracted_final_answer_match.group(1).strip() if extracted_final_answer_match else "None"

        # Parse reasoning
        reasoning_match = re.search(r"reasoning: (.+?)(?=\n\w+:|$)", grading_response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Parse correctness
        correct_match = re.search(r"correct: (yes|no)", grading_response)
        correctness = correct_match.group(1) if correct_match else "no"

        # Parse confidence
        confidence_match = re.search(r"confidence: (\d+)", grading_response)
        confidence = int(confidence_match.group(1)) if confidence_match else 100

        return {
            "extracted_final_answer": extracted_final_answer,
            "reasoning": reasoning,
            "correctness": correctness,
            "confidence": confidence
        }

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            question_text = row['question']
            correct_answer = row['answer']
            
            # Format the prompt using the original format_message function
            formatted_prompt = format_message(row)
            
            prompt_messages = [
                sampler._pack_message(content=formatted_prompt, role="user")
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            
            # Grade the sample and get parsed results
            grade_result = self.grade_sample(question_text, correct_answer, response_text)
            
            # Metrics based on grading response
            score = grade_result["correctness"] == "yes"

            # Create HTML for each sample result with additional grader information
            extra_info = {
                "Grader Reasoning": grade_result["reasoning"],
                "Grader Extracted Confidence": f"{grade_result['confidence']}%",
                "Grader Correctness": grade_result["correctness"],
            }

            model_response = [dict(content=response_text, role="assistant", type="text")]
            if "extra_convo" in sampler_response.response_metadata:
                model_response = sampler_response.response_metadata.pop("extra_convo") + model_response
            
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=model_response,
                score=score,
                correct_answer=correct_answer,
                extracted_answer=grade_result["extracted_final_answer"],
                extra_info=extra_info,
            )
            convo = actual_queried_prompt_messages + model_response
            grade_result["id"] = row["id"]
            sampler_response.response_metadata.pop("usage", None)
            grade_result["response_metadata"] = sampler_response.response_metadata
            
            return SingleEvalResult(
                html=html, 
                score=score, 
                convo=convo, 
                metrics={
                    "is_correct": score,
                    "confidence": grade_result["confidence"],
                },
                example_level_metadata=grade_result
            )

        # Run evaluation and collect results
        results = common.map_with_progress(fn, self.examples, num_threads=self.n_threads)

        calibration_error = calib_err(
            np.array([result.metrics["confidence"] for result in results]),
            np.array([result.metrics["is_correct"] for result in results]),
            p='2',
            beta=100
        )
        # Aggregate metrics
        aggregate_metrics = {
            "accuracy": sum(result.metrics["is_correct"] for result in results) / len(results),
            "calibration_error": calibration_error,
        }
        confidence_half_width = round(1.96 * math.sqrt(aggregate_metrics["accuracy"] * (100 - aggregate_metrics["accuracy"]) / len(results)), 2)
        aggregate_metrics["confidence_half_width"] = confidence_half_width

        print("AGGREGATE METRICS") 
        print(aggregate_metrics) 
        print("##################")

        output_d = {
            "accuracy": aggregate_metrics["accuracy"],
            "calibration_error": aggregate_metrics["calibration_error"],
            "confidence_half_width": aggregate_metrics["confidence_half_width"],
        }
        
        print(f"Accuracy: {output_d['accuracy']:.3f}")
        print(f"Calibration Error: {output_d['calibration_error']:.1f}%")
        print(f"Confidence Half Width: {output_d['confidence_half_width']:.1f}%")
        
        # Use aggregate_results but add our custom metrics
        eval_result = common.aggregate_results(results)
        
        # Add calibration error and confidence_half_width to the final metrics
        eval_result.metrics["calibration_error"] = aggregate_metrics["calibration_error"]
        eval_result.metrics["confidence_half_width"] = aggregate_metrics["confidence_half_width"]
        
        return eval_result