import time

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoModelForSeq2SeqLM

from pipelines.lightning_base_class import FLEdgeLightningBase
from pipelines.flan_t5_dialogue_summarization.utils import compute_rogue


class FlanT5Lightning(FLEdgeLightningBase):

    def __init__(self, tokenizer, *args, **kwargs):
        super(FlanT5Lightning, self).__init__(*args, **kwargs)
        model_name = self.config["model_name"]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.tokenizer = tokenizer
        self.optim = AdamW(self.model.parameters(), lr=0.0001)

        # Scoring
        self.rouge_score = ROUGEScore()

    def forward(self, batch, stage="train"):
        """
        We use the forward method for training. generation is done via the models 'generate' function.
        :param batch: Data batch of shape input_ids, attention_mask, labels (dict).
        :param stage: Stage indicator. Not used in this particular model. (str)
        """
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        return out

    def training_step(self, batch, batch_i):
        start_time = time.perf_counter()
        out = self.forward(batch, stage="train")
        self.log("timing/train/forward_time", time.perf_counter() - start_time)
        # Loss is calculated inside the transformer. Therefore, measurement is impractical.
        rogue_start_time = time.perf_counter()
        duration = time.perf_counter() - rogue_start_time

        self.log("timing/train/rogue_calc_time", duration)
        self.log("train/loss", out["loss"])
        self.log("train/accuracy", out["loss"] / len(batch))

        res = {"loss": out["loss"]}
        return res["loss"]

    def validation_step(self, batch, batch_i):
        start_time = time.perf_counter()
        out = self.model.generate(batch["input_ids"], max_length=95)
        self.log("timing/val/generate_time", time.perf_counter() - start_time)

        # Generate the adequacy measure and calculate timings.
        start_time = time.perf_counter()
        rouge_scores = compute_rogue(logits=out, labels=batch["labels"], tokenizer=self.tokenizer)
        self.log("timing/val/rouge_calc_time", time.perf_counter() - start_time)

        for k, v in rouge_scores.items():
            self.log(f"val/{k}", v)

        return rouge_scores["rouge1"]

    def test_step(self, batch, batch_i):
        start_time = time.perf_counter()
        out = self.model.generate(batch["input_ids"], max_length=95)
        self.log("timing/test/generate_time", time.perf_counter() - start_time)

        # Generate the adequacy measure and calculate timings.
        start_time = time.perf_counter()
        rouge_scores = compute_rogue(logits=out, labels=batch["labels"], tokenizer=self.tokenizer)
        self.log("timing/test/rouge_calc_time", time.perf_counter() - start_time)

        for k, v in rouge_scores.items():
            self.log(f"test/{k}", v)

        return rouge_scores

    def configure_optimizers(self):
        scheduler = StepLR(
            optimizer=self.optim,
            step_size=1,
            gamma=0.85
        )

        return {"optimizer": self.optim, "lr_scheduler": scheduler}

