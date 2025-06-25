import argparse
import json
import numpy as np
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import warnings

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(args):
    cider = Cider()
    spice = Spice()
    meteor = Meteor()
    rouge = Rouge()
    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    preds = read_jsonl(args.pred_file)
    refs = read_jsonl(args.ref_file)

    pred_map = {p['audio_path']: p['model_response'] for p in preds}
    ref_map = {r['audio_path']: r['model_response'] for r in refs}

    metrics = defaultdict(list)

    for audio_path in pred_map:
        if audio_path not in ref_map:
            continue

        candidate = pred_map[audio_path]
        reference = ref_map[audio_path]

        # METEOR
        m_score, _ = meteor.compute_score({0: [reference]}, {0: [candidate]})
        metrics['METEOR'].append(m_score)

        # ROUGE-L
        r_score, _ = rouge.compute_score({0: [reference]}, {0: [candidate]})
        metrics['ROUGE-L'].append(r_score)

        # CIDEr
        c_score, _ = cider.compute_score({0: [reference]}, {0: [candidate]})
        metrics['CIDEr'].append(c_score)

        # SPICE
        try:
            s_score, _ = spice.compute_score({0: [reference]}, {0: [candidate]})
            metrics['SPICE'].append(s_score)
        except:
            metrics['SPICE'].append(0)

        # SPIDEr
        metrics['SPIDEr'].append((c_score + s_score) / 2)

        # Sentence-BERT
        emb1 = sbert_model.encode(candidate, convert_to_tensor=True)
        emb2 = sbert_model.encode(reference, convert_to_tensor=True)
        metrics['Sentence-BERT'].append(util.pytorch_cos_sim(emb1, emb2).item())

    print("Average Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True, help="Path to predictions JSONL file")
    parser.add_argument('--ref_file', type=str, required=True, help="Path to references JSONL file")
    args = parser.parse_args()
    main(args)
