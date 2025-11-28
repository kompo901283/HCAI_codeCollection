import json
import os

def convert_one(data):
    lines = []
    for sent in data.get("parsed", []):
        text = sent.get("input", "")
        lines.append(f"# text = {text}")
        labels = sent.get("labels", [])

        for idx, tok in enumerate(labels, start=1):
            tid = idx
            form = tok.get("text", "_")
            lemma = tok.get("lemma", "_")
            upos = tok.get("upos", "_")
            xpos = tok.get("xpos", "_")
            feats = tok.get("feats", "_") or "_"
            head = tok.get("head", "_")
            deprel = tok.get("deprel", "_")
            deps = tok.get("deps", "_")
            misc = tok.get("misc", "_") or "_"

            lines.append(f"{tid}\t {form}\t {lemma}\t {upos}\t {xpos}\t {feats}\t {head}\t {deprel}\t{deps}\t {misc}")

    return "\n".join(lines)

def convert_jsonl_to_conllu(jsonl_path: str, conllu_path: str):
    with open(jsonl_path,"r", encoding="utf-8") as fin, open(conllu_path,"w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            conllu_text = convert_one(data)
            fout.write(conllu_text)
            fout.write("\n\n")

def run():
    converted_files = set(os.listdir("Labeled_ConLL"))
    rawJson_files = os.listdir("Labeled_Unzipped")
    for fname in rawJson_files:
        if fname.endswith(".jsonl") and fname.replace(".jsonl", ".conllu") not in converted_files:
            src = os.path.join("Labeled_Unzipped", fname)
            output_name = fname.rsplit(".")[0] + ".conll"
            dst = os.path.join("Labeled_ConLL", output_name)
            convert_jsonl_to_conllu(src, dst)

run()



