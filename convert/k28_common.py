"""Shared helpers for the K28 conversion scripts (fetch, golden, forward)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_JSON = ROOT / "data/k28/models.json"
K28_DIR = ROOT / "models/k28"

PILOT = [
    "daekeun-ml/koelectra-small-v3-nsmc",
    "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L4-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
    "dslim/bert-base-NER",
]


# K28.5: RoBERTa, XLM-R and DistilBERT, every task and every quirk of the three rows.
PILOT_K285 = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "cross-encoder/nli-distilroberta-base",
    "sentence-transformers/all-distilroberta-v1",
    "OpenMed/OpenMed-NER-OrganismDetect-TinyMed-82M",
    "cross-encoder/stsb-distilroberta-base",
    "qilowoq/mmarco-mMiniLMv2-L12-H384-v1-en-ru",
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    "d0rj/e5-small-en-ru",
    "ukr-models/uk-ner",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "typeform/distilbert-base-uncased-mnli",
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-65M",
    "Amdestya/ce-cat-distilbert",
    "emrecan/distilbert-base-turkish-cased-allnli_tr",
]


# K28.6: DeBERTa-v2/v3 and ModernBERT, every task and every quirk of the two rows.
PILOT_K286 = [
    "protectai/deberta-v3-base-prompt-injection-v2",
    "cross-encoder/nli-deberta-v3-small",
    "xushijie/polyBERT",
    "OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M",
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "sheltron-ai/prompt-guard-68m",
    "Horizon-Labs/multilingual-zeroshot-small",
    "ibm-granite/granite-embedding-small-english-r2",
    "OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-149M",
    "hotchpotch/japanese-reranker-xsmall-v2",
    "ibm-granite/granite-embedding-reranker-english-r2",
]


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_entry(model_id: str) -> dict:
    for entry in json.loads(MODELS_JSON.read_text())["models"]:
        if entry["id"] == model_id:
            return entry
    raise KeyError(f"{model_id} is not in data/k28/models.json")


def model_dir(model_id: str) -> Path:
    return K28_DIR / slug(model_id)


def repo_dir(model_id: str) -> Path:
    return model_dir(model_id) / "repo"


def weight_file(entry: dict) -> str:
    """The checkpoint file the pipeline reads: safetensors when the repo has it."""
    files = entry["weightFiles"]
    return "model.safetensors" if "model.safetensors" in files else files[0]


def wordpiece_ids() -> list[str]:
    """Models whose tokenizer.json has a WordPiece model, plus those with only vocab.txt."""
    out = []
    for e in json.loads(MODELS_JSON.read_text())["models"]:
        tj = e.get("tokenizerJson") or {}
        if tj.get("model") == "WordPiece" or e["tokenizerFiles"] == ["vocab.txt"]:
            out.append(e["id"])
    return out


def nonwordpiece_ids() -> list[str]:
    """The models of the list that wordpiece_ids() leaves out (K28.3b)."""
    wp = set(wordpiece_ids())
    return [e["id"] for e in json.loads(MODELS_JSON.read_text())["models"] if e["id"] not in wp]


def tokenizer_json(model_id: str) -> dict:
    """The tokenizer.json of a model: the converted one next to the repo, else the repo's own."""
    own = model_dir(model_id) / "tokenizer.json"
    return json.loads((own if own.exists() else repo_dir(model_id) / "tokenizer.json").read_text())


def single_template(tj: dict) -> list:
    """The single template of a tokenizer.json in order: special token ids, None for the text."""
    post = tj.get("post_processor")
    if post is None:
        return [None]
    if post["type"] == "TemplateProcessing":
        out = []
        for item in post["single"]:
            if "Sequence" in item:
                out.append(None)
            else:
                out.extend(post["special_tokens"][item["SpecialToken"]["id"]]["ids"])
        return out
    if post["type"] == "RobertaProcessing":
        return [post["cls"][1], None, post["sep"][1]]
    if post["type"] == "ByteLevel":
        return [None]
    raise ValueError(f"post_processor {post['type']} is not read")


def position_offset(cfg, model_id: str) -> int:
    """Rows of the position table skipped before position 0 (RoBERTa rule, K28.5, template form K28.3b).

    HF numbers positions as pad_token_id + cumulative count of tokens that are not
    pad_token_id. The first token of a text is the first piece of the single template of
    tokenizer.json. Normally it is no pad token and position i sits at row i + pad_token_id + 1.
    When it has the pad id (d0rj/e5-small-en-ru: <s> and <pad> are both 0), it counts as padding,
    gets row pad_token_id, and token i sits at row i + pad_token_id. A later special token of the
    template with the pad id shifts every position after it: that raises.
    """
    if cfg.model_type in ("roberta", "xlm-roberta"):
        pad = int(cfg.pad_token_id)
        template = single_template(tokenizer_json(model_id))
        if any(t is not None and t == pad for t in template[1:]):
            raise ValueError(f"pad_token_id {pad} is the id of a later special token of the single template")
        return pad if template[0] == pad else pad + 1
    return 0


def resolve_ids(ids: list[str], pilot: bool, wordpiece: bool = False,
                k285: bool = False, nonwordpiece: bool = False, k286: bool = False) -> list[str]:
    if nonwordpiece:
        return nonwordpiece_ids()
    if wordpiece:
        return wordpiece_ids()
    if k286:
        return list(PILOT_K286) + [i for i in ids if i not in PILOT_K286]
    if k285:
        return list(PILOT_K285) + [i for i in ids if i not in PILOT_K285]
    if pilot:
        return list(PILOT) + [i for i in ids if i not in PILOT]
    if not ids:
        raise SystemExit("give model ids or --pilot")
    return ids
