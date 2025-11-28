import os, gzip, json
import stanza
from pathlib import Path

stanza.download("nb")
nlp = stanza.Pipeline("nb", processors="tokenize,pos,lemma,depparse", use_gpu=True)

input_root = "maalfrid_2021"
output_root = "maalfrid_labeled"
os.makedirs(output_root, exist_ok=True)

dataDir = Path(output_root)
labeledFiles = {p.name for p in dataDir.iterdir()}

for fname in os.listdir(input_root):
    if not (fname.endswith(".jsonl.gz") and "nob" in fname) or fname in labeledFiles:
        continue

    in_path = os.path.join(input_root, fname)
    out_path = os.path.join(output_root, fname)

    with gzip.open(in_path, "rt", encoding="utf-8") as fin, \
         gzip.open(out_path, "wt", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)
            new_blocks = []

            for text in obj["fulltext"]:
                doc = nlp(text)
                sent_labels = []
                for sent in doc.sentences:
                    for word in sent.words:
                        sent_labels.append({
                            "id": word.id,
                            "text": word.text,
                            "lemma": word.lemma,
                            "upos": word.upos,
                            "xpos": word.xpos,
                            "feats": word.feats,
                            "head": word.head,
                            "deprel": word.deprel,
                            "misc": word.misc,
                        })
                new_blocks.append({
                    "input": text,
                    "labels": sent_labels
                })

            obj["parsed"] = new_blocks
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
