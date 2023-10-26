import nltk
import evaluate
import numpy as np
import torch
import transformers
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
metric = evaluate.load("rouge")


# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_rogue(logits: torch.Tensor, labels: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer):
    if isinstance(logits, tuple):
        logits = logits[0]

    tk_logits = tokenizer.batch_decode(
        sequences=logits,
        skip_special_tokens=True
    )

    labels = labels.cpu().numpy()

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    tk_labels = tokenizer.batch_decode(
        sequences=labels,
        skip_special_tokens=True
    )

    tk_logits, tk_labels = postprocess_text(tk_logits, tk_labels)

    result = metric.compute(predictions=tk_logits, references=tk_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}

    logits = logits.cpu().numpy()
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in logits]
    result["gen_len"] = np.mean(prediction_lens)
    return result

