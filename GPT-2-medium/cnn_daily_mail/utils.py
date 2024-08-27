from dataclasses import dataclass
import os
import json
import bitsandbytes as bnb
import copy
import pandas as pd
import torch
import transformers
from evaluate import load as load_metric
import numpy as np
import pandas as pd

from typing import Dict, List, Sequence, Tuple, Union
from tqdm.notebook import tqdm
from dataclasses import dataclass

from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedTokenizer, 
    PreTrainedModel, 
    GenerationConfig
)



import logging
import pandas as pd

IGNORE_INDEX = -100


def compute_bleu_score(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Calculate the BLEU score and n-gram precision scores between predicted and true sequences.

    Arguments:
    -----------
        - y_true (List[str]): A list of true text sequences.
        - y_pred (List[str]): A list of predicted text sequences.

    Returns:
    --------
        - Dict[str, float]
            A dictionary containing BLEU score and n-gram precision scores.
            Example:
            {
                "bleu_score": 0.85,
                "1-gram precision": 0.90,
                "2-gram precision": 0.85,
                "3-gram precision": 0.78,
                "4-gram precision": 0.72
            }
    """
    
    # Load the BLEU metric
    blue_metric = load_metric("bleu")

    # Compute BLEU score and other metrics
    result = blue_metric.compute(predictions=y_pred, references=y_true)

    # Extract BLEU score
    bleu_score = {"bleu_score": result['bleu']}

    # Extract n-gram precision scores
    n_gram_precision = {f"{n + 1}-gram precision": v for n, v in enumerate(result["precisions"])}

    # Combine BLEU and n-gram precision scores into a dictionary
    bleu_score.update(n_gram_precision)

    return bleu_score


def compute_rouge_score(y_true: List[str], y_pred: List[str]) -> dict:
    """
    Compute the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score.

    This function computes the ROUGE score, which is a metric commonly used in natural language
    processing to evaluate the quality of text generation models. ROUGE measures the similarity
    between the predicted text (y_pred) and the ground truth or reference text (y_true).

    Arguments:
    ----------
        - y_true (List[str]): A list of reference text strings.
        - y_pred (List[str]): A list of predicted text strings.

    Returns:
    -------
        - dict: A dictionary containing ROUGE scores.

    Example:
    --------
        y_true = ["reference text 1", "reference text 2"]
        y_pred = ["predicted text 1", "predicted text 2"]
        rouge_scores = compute_rouge_score(y_true, y_pred)
    """

    # Load the ROUGE metric from the datasets library
    rouge_metric = load_metric("rouge")
    
    # Compute ROUGE scores
    result = rouge_metric.compute(predictions=y_pred, references=y_true)
    
    return result


def compute_metrics(y_true: List[str], y_pred: List[str], 
                    metrics_to_compute : Tuple[str]= ("bleu", "rouge"), 
                    **kwargs):
    """
    Compute evaluation metrics for a given set of true and predicted text sequences.

    Arguments:
    ----------
        - y_true (List[str]): A list of true text sequences.
        - y_pred (List[str]): A list of predicted text sequences. 
        - metrics_to_compute (Tuple[str]): The name of the metrics to compute. Options: ("bleu", "rouge").

    Returns:
    --------
        - pd.DataFrame: A DataFrame containing computed evaluation metrics, including BLEU, ROUGE and cosine similarity.
            The DataFrame has a single row with the metrics as columns.

    Example:
    --------
        y_true = ["reference text 1", "reference text 2"]
        y_pred = ["predicted text 1", "predicted text 2"]
        metrics_df = compute_metrics(y_true, y_pred, metrics_to_compute=("bleu", "rouge")
    """
    scores = {}
    for metric in metrics_to_compute:
        if metric == "bleu":
            
            # Compute BLEU scores
            print("Computing BLEU scores")
            bleu_scores = compute_bleu_score(y_true, y_pred)
            scores.update(bleu_scores)
            
        elif metric == "rouge":
            
            # Compute ROUGE scores
            print("Computing ROUGE scores")
            rouge_scores = compute_rouge_score(y_true, y_pred)
            scores.update(rouge_scores)

    # Create a DataFrame with the scores
    metrics_df = pd.DataFrame(scores, index=[0])

    return metrics_df        

class LLMInference:
    """
    A class for performing inference with a quantized language model, capable of loading models with LoRA adapters,
    generating responses, and computing metrics.
    """

    def __init__(
        self,
        fine_tuned_model: Union[AutoModelForCausalLM],
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        """
        Initializes the LLMInference with the specified configuration for loading and quantizing models.
        Here the parameters used in training are used to load de base model.

        Arguments:
        ----------
            - fine_tuned_model (Union[AutoModelForCausalLM, AutoPeftModelForCausalLM], optional): A pytorch pre-trained model.
            - tokenizer (transformers.PreTrainedTokenizer): A tokenizer for the specified language model.
        """

        self.model = fine_tuned_model
        self.tokenizer = tokenizer

    def generate(
        self,
        dataset,
        batch_size: int,
        generation_config: GenerationConfig,
        source_max_len: int = 1024,
        padding_side="left",
    ) -> Tuple[List[str], List[str], dict]:
        """
        Generates responses for the input prompts in the dataset using a specified peft model.

        This function processes the dataset in batches, generates responses using the peft model,
        and returns the generated responses along with the expected labels.

        Arguments:
        ----------
            - dataset (datasets.Dataset): A datasets.Dataset object containing 'input' and 'output' keys.
                'input' is a list of prompts, and 'output' is a list of expected responses.
            - batch_size (int): The size of each batch for processing.
            - generation_config (GenerationConfig): The configuration for generating responses.
            - source_max_len (int, optional): The maximum length of the source text. Defaults to 1024.
            - padding_side (str, optional): The side to pad the source text. Defaults to "left".
            - response_indicator (str, optional): The string used to indicate the start of a response. Defaults to "### Response:\n".

        Returns:
        --------
            - Tuple[List[str], List[str]]: A tuple containing two lists:
                - The first list contains the generated responses.
                - The second list contains the expected labels (ground truth).
        """

        inputs = []
        label = []

        for inp, df in dataset.to_pandas().groupby("input"):

            inputs.append(inp)
            label.append(df["output"].tolist())


        # Split the input data into batches
        sources = [f"{self.tokenizer.bos_token}{example}" for example in inputs]
        prompt = [sources[i : i + batch_size] for i in range(0, len(sources), batch_size)]
        output = []

        # Iterate over each batch and generate responses
        for batch in tqdm(prompt):
            with torch.no_grad():
                # Tokenize the source texts
                tokens = self.tokenizer(batch, max_length=source_max_len, truncation=True, add_special_tokens=False)

                # pad the source text to the left
                if padding_side == "left":
                    tokens = [torch.tensor(tk[::-1]) for tk in tokens["input_ids"]]

                    # Pad sequences to the longest one in the batch
                    input_ids = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])

                elif padding_side == "right":
                    tokens = [torch.tensor(tk) for tk in tokens["input_ids"]]

                    # Pad sequences to the longest one in the batch
                    input_ids = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)

                # Prepare the final data dictionary
                data_dict = {
                    "input_ids": input_ids.to("cuda"),
                    "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).to("cuda"),  # Create attention mask
                }

                # Generate responses using the model with the provided generation configuration
                response = self.model.generate(**data_dict, generation_config=generation_config, pad_token_id=self.tokenizer.pad_token_id).to("cpu")

            # Decode the generated tokens to strings and append to the output list
            output.append([self.tokenizer.decode(out[input_ids.shape[1]:], skip_special_tokens=True) for out in response])
            #output.append([self.tokenizer.decode(out, skip_special_tokens=True) for out in response])

        # Post-process the generated responses to remove any additional formatting
        # output = [k[k.find(response_indicator) + len(response_indicator) :].strip() for k in [j for i in output for j in i]]
        output = [k for k in [j for i in output for j in i]]

        # Return the generated responses and the expected labels
        return output, label, inputs

    def save_predictions_and_metrics(self):
        """
        Saves the generated predictions and computed metrics to the specified artifacts or to a local directory.

        This method checks if the 'predictions_and_metrics_artifact_name' and 'predictions_and_metrics_artifact_version'
        attributes are set. If they are, it saves the metrics and predictions as artifacts. Otherwise, it saves them
        to a local directory.
        """

        path = "./predictions_and_metric"

        if not os.path.exists(path):
            os.mkdir(path)
            
        self.metrics.to_csv(f"{path}/metrics.csv", index=False)
        self.test_dataset.to_csv(f"{path}/predictions.csv", index=False)

    def make_predictions_and_compute_metrics(
        self,
        dataset,
        batch_size: int,
        source_max_len: int,
        #response_indicator: str,
        padding_side: str,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.2,
        top_p=95,
        top_k=40,
        repetition_penalty=1.1,
        generation_config_kwargs: dict = {},
    ):
        """
        Generates predictions for the given dataset and computes metrics to evaluate the model's performance.

        This method uses the 'generate' method to produce responses and then computes various metrics such as BLEU and ROUGE.
        This method also save the results and predictions on a artifact or local folder.

        Arguments:
        ----------
            - dataset (datasets.Dataset): The dataset containing input prompts and expected responses.
            - batch_size (int): The size of each batch for processing.
            - source_max_len (int): The maximum length of the source text.
            - response_indicator (str): The string used to indicate the start of a response.
            - max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 1024.
            - do_sample (bool, optional): Whether to use sampling for generation. Defaults to True.
            - temperature (float, optional): The temperature for sampling. Defaults to 0.2.
            - top_p (int, optional): The nucleus sampling probability. Defaults to 95.
            - top_k (int, optional): The top-k sampling probability. Defaults to 40.
            - repetition_penalty (float, optional): The penalty for repetition. Defaults to 1.1.
            - generation_config_kwargs (dict, optional): Additional keyword arguments for the generation configuration.
        """

        if do_sample:
            self.generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **generation_config_kwargs,
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                **generation_config_kwargs,
            )

        y_pred, y_true, inputs = self.generate(
            dataset=dataset,
            batch_size=batch_size,
            generation_config=self.generation_config,
            source_max_len=source_max_len,
            padding_side=padding_side,
            #response_indicator=response_indicator,
        )
        
        # compute metrics
        self.metrics = compute_metrics(y_true, y_pred, metrics_to_compute=("bleu", "rouge")).round(3)

        # store predictions
        y_pred = [[pred]*len(true) for pred, true in zip(y_pred, y_true)]
        y_pred = [j for i in y_pred for j in i]
        
        inputs = [[inp]*len(true) for inp, true in zip(inputs, y_true)]
        inputs = [j for i in inputs for j in i]

        y_true = [j for i in y_true for j in i]

        self.test_dataset = pd.DataFrame({"input": inputs, "output": y_true, "prediction":y_pred})

        # save predictions and metrics
        self.save_predictions_and_metrics()
