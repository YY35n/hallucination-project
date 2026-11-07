#!/usr/bin/env python3
"""
DTSC 710 project - Detecting and Reducing Hallucinations in LLMs
Runs the four conditions from the proposal on the same model and questions:

  A  direct answer
  B  A + "say I don't know if unsure"
  C  B + top-k retrieved Wikipedia passages
  D  second call checks C's draft against the same passages (KEEP / CORRECT / WITHDRAW)

Everything is logged per question so errors can be traced later.

Quick start (macOS):
  1. Install Ollama (https://ollama.com) and run:   ollama pull qwen2.5:3b
  2. pip install rank-bm25 requests
  3. python run_experiment.py --n 300
Outputs go to ./results/
"""
import argparse, json, os, random, re, string, time, csv, math
from collections import Counter, defaultdict

import requests
from rank_bm25 import BM25Okapi

NQ_URL = ("https://raw.githubusercontent.com/google-research-datasets/"
          "natural-questions/master/nq_open/NQ-open.dev.jsonl")
WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "DTSC710-hallucination-course-project/1.0 (academic use)"}

# --------------------------------------------------------------------------- data

TIME_WORDS = re.compile(r"\b(current|currently|now|latest|this year|as of|recent|recently|today|newest)\b", re.I)


def load_questions(path, n, seed, max_ans_tokens=5):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(NQ_URL, timeout=60)
        r.raise_for_status()
        open(path, "w").write(r.text)
    rows = [json.loads(l) for l in open(path)]
    total = len(rows)
    kept, dropped = [], Counter()
    for i, r in enumerate(rows):
        q, ans = r["question"], r["answer"]
        if TIME_WORDS.search(q):
            dropped["time-sensitive"] += 1
            continue
        if min(len(a.split()) for a in ans) > max_ans_tokens:
            dropped["long answer"] += 1
            continue
        if any(a.lower() in ("yes", "no") for a in ans):
            dropped["yes/no"] += 1
            continue
        kept.append({"qid": f"nq-dev-{i}", "question": q, "answers": ans})
    random.Random(seed).shuffle(kept)
    stats = {"total_dev": total, "after_filter": len(kept), "dropped": dict(dropped), "sampled": min(n, len(kept))}
    return kept[:n], stats


# --------------------------------------------------------------------------- corpus

# polite throttle + backoff so the Wikipedia API does not rate-limit us (HTTP 429/503)
_WIKI_MIN_INTERVAL = 1.2   # seconds between requests (~0.8 req/s, safe for the search endpoint)
_wiki_last = [0.0]


def wiki_get(params, tries=6):
    params = dict(params, maxlag=5)  # ask the API to shed load gracefully when replicas lag
    for attempt in range(tries):
        wait = _WIKI_MIN_INTERVAL - (time.time() - _wiki_last[0])
        if wait > 0:
            time.sleep(wait)
        r = requests.get(WIKI_API, params=params, headers=UA, timeout=30)
        _wiki_last[0] = time.time()
        if r.status_code in (429, 503):  # rate-limited or maxlag exceeded
            back = float(r.headers.get("Retry-After", 2 ** attempt))
            time.sleep(min(back, 30))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def wiki_search(query, limit):
    p = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
    r = wiki_get(p)
    return [h["title"] for h in r.json()["query"]["search"]]


def wiki_text(title):
    p = {"action": "query", "prop": "extracts|revisions", "rvprop": "ids|timestamp", "explaintext": 1,
         "redirects": 1, "titles": title, "format": "json"}
    r = wiki_get(p)
    page = next(iter(r.json()["query"]["pages"].values()))
    rev = (page.get("revisions") or [{}])[0]
    return page.get("extract", ""), rev.get("revid"), rev.get("timestamp")


def chunk(text, size=100):
    text = re.sub(r"\n==+ (See also|References|External links|Further reading|Notes)[\s\S]*", "", text)
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size) if len(words[i:i + size]) > 20]


def build_corpus(questions, out_dir, pages_per_q, fake=False):
    """Closed corpus: pages found by searching Wikipedia with the question text
    (never with the gold answer), pooled over all questions, split into 100-word chunks."""
    path = os.path.join(out_dir, "corpus.jsonl")
    meta_path = os.path.join(out_dir, "corpus_meta.json")
    if os.path.exists(path):
        return [json.loads(l) for l in open(path)], json.load(open(meta_path))
    pages = {}
    t0 = time.time()
    for i, q in enumerate(questions):
        if fake:  # offline smoke test only
            pages[f"Fake page {i}"] = (f"Some text about {q['question']}. The answer is {q['answers'][0]}. " * 15, 0, "")
            continue
        try:
            for t in wiki_search(q["question"], pages_per_q):
                if t not in pages:
                    pages[t] = wiki_text(t)
        except Exception as e:
            print("  wiki error:", q["qid"], e)
        if (i + 1) % 25 == 0:
            print(f"  corpus: {i+1}/{len(questions)} questions, {len(pages)} pages")
    passages = []
    for title, (txt, revid, ts) in pages.items():
        for j, c in enumerate(chunk(txt)):
            passages.append({"pid": f"{title}#{j}", "title": title, "text": c, "revid": revid})
    with open(path, "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    meta = {"pages": len(pages), "passages": len(passages), "built": time.strftime("%Y-%m-%d"),
            "build_seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(meta_path, "w"), indent=2)
    return passages, meta


TOKEN_RE = re.compile(r"\w+")
STOP = set("the a an of in on at to for is was were are be by with and or who what when where which how did does do".split())


def tok(s):
    return [w for w in TOKEN_RE.findall(s.lower()) if w not in STOP]


class Retriever:
    def __init__(self, passages):
        self.p = passages
        self.bm25 = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in passages])

    def search(self, q, k):
        t0 = time.perf_counter()
        scores = self.bm25.get_scores(tok(q))
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.p[i] for i in idx], (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- model

class Ollama:
    def __init__(self, model, host):
        self.model, self.url = model, host.rstrip("/") + "/api/chat"

    def __call__(self, prompt, max_tokens):
        r = requests.post(self.url, json={
            "model": self.model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens}}, timeout=600)
        r.raise_for_status()
        j = r.json()
        return j["message"]["content"].strip(), j.get("prompt_eval_count", 0), j.get("eval_count", 0)


class Dummy:  # smoke test only, NOT for results
    def __init__(self):
        self.r = random.Random(0)

    def __call__(self, prompt, max_tokens):
        if "DECISION" in prompt:
            out = self.r.choice(["DECISION: KEEP\nANSWER: x\nREASON: ok", "DECISION: WITHDRAW\nANSWER: \nREASON: no"])
        else:
            out = self.r.choice(["I don't know", "Paris", "1972"])
        return out, len(prompt.split()), len(out.split())


# --------------------------------------------------------------------------- prompts

P_A = ("Answer the question with a short answer of a few words. Do not explain.\n\n"
       "Question: {q}\nAnswer:")
P_B = ("Answer the question with a short answer of a few words. Do not explain.\n"
       "If you are not sure of the answer, reply exactly: I don't know\n\n"
       "Question: {q}\nAnswer:")
P_C = ("Answer the question with a short answer of a few words. Do not explain.\n"
       "Use the passages below if they help. "
       "If you are not sure of the answer, reply exactly: I don't know\n\n"
       "{ctx}\n\nQuestion: {q}\nAnswer:")
P_D = ("You are checking a short answer to a question against some Wikipedia passages.\n\n"
       "{ctx}\n\nQuestion: {q}\nDraft answer: {draft}\n\n"
       "Decide one of:\n"
       "KEEP - the passages support the draft answer.\n"
       "CORRECT - the passages show a different answer; give that answer.\n"
       "WITHDRAW - the passages contradict the draft, or give no support and you cannot tell the answer.\n\n"
       "Example:\nDECISION: CORRECT\nANSWER: George Washington\nREASON: Passage [2] says Washington was the first president.\n\n"
       "Reply in exactly this format:\nDECISION: <KEEP|CORRECT|WITHDRAW>\nANSWER: <short answer>\nREASON: <one sentence>")


def ctx_block(passages):
    return "Passages:\n" + "\n".join(f"[{i+1}] ({p['title']}) {p['text']}" for i, p in enumerate(passages))


def clean_answer(s):
    s = s.strip().split("\n")[0]
    s = re.sub(r"^(answer|final answer)\s*:\s*", "", s, flags=re.I)
    return s.strip().strip('"').rstrip(".").strip()


def parse_checker(out):
    m = re.search(r"DECISION\s*:\s*\**\s*(KEEP|CORRECT|WITHDRAW)", out, re.I)
    a = re.search(r"ANSWER\s*:\s*(.*)", out, re.I)
    rs = re.search(r"REASON\s*:\s*(.*)", out, re.I)
    dec = m.group(1).upper() if m else None
    return dec, clean_answer(a.group(1)) if a else "", rs.group(1).strip() if rs else ""


# --------------------------------------------------------------------------- scoring

ABSTAIN = re.compile(r"(i don'?t know|i do not know|not sure|unknown|cannot (be )?determin|can'?t determin|"
                     r"no (information|answer)|not (provided|mentioned|enough)|unable to|i'?m not certain)", re.I)

# alias rules fixed before looking at outputs
ALIASES = {"nyc": "new york city", "us": "united states", "usa": "united states", "u s": "united states",
           "uk": "united kingdom", "ussr": "soviet union", "un": "united nations"}
NUMS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def norm(s):
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(NUMS.get(w, w) for w in s.split())
    return ALIASES.get(s, s)


def is_abstain(ans):
    return (not ans.strip()) or bool(ABSTAIN.search(ans))


def is_correct(ans, golds):
    p = norm(ans)
    if not p:
        return False
    for g in golds:
        g = norm(g)
        if not g:
            continue
        if p == g or re.search(r"\b" + re.escape(g) + r"\b", p):
            return True
    return False


def label(ans, golds):
    if is_abstain(ans):
        return "abstain"
    return "correct" if is_correct(ans, golds) else "incorrect"


def gold_in_passages(golds, passages):
    text = norm(" ".join(p["text"] for p in passages))
    return any(norm(g) and re.search(r"\b" + re.escape(norm(g)) + r"\b", text) for g in golds)


# --------------------------------------------------------------------------- run

def run_one(q, llm, ret, k):
    rec = {"qid": q["qid"], "question": q["question"], "answers": q["answers"]}
    out, pi, po = llm(P_A.format(q=q["question"]), 32)
    rec["A"] = {"raw": out, "answer": clean_answer(out), "calls": 1, "tok_in": pi, "tok_out": po}
    out, pi, po = llm(P_B.format(q=q["question"]), 32)
    rec["B"] = {"raw": out, "answer": clean_answer(out), "calls": 1, "tok_in": pi, "tok_out": po}
    passages, ms = ret.search(q["question"], k)
    ctx = ctx_block(passages)
    rec["passage_ids"] = [p["pid"] for p in passages]
    rec["retrieval_ms"] = ms
    rec["gold_in_topk"] = gold_in_passages(q["answers"], passages)
    out, pi, po = llm(P_C.format(q=q["question"], ctx=ctx), 32)
    draft = clean_answer(out)
    rec["C"] = {"raw": out, "answer": draft, "calls": 1, "tok_in": pi, "tok_out": po}
    out2, pi2, po2 = llm(P_D.format(q=q["question"], ctx=ctx, draft=draft or "I don't know"), 96)
    dec, new, reason = parse_checker(out2)
    parsed = dec is not None
    dec = dec or "KEEP"  # unparseable -> keep the draft, counted separately
    final = draft if dec == "KEEP" else (new if dec == "CORRECT" and new else "I don't know")
    rec["D"] = {"raw": out2, "decision": dec, "parsed": parsed, "reason": reason, "answer": final,
                "calls": 2, "tok_in": pi + pi2, "tok_out": po + po2}
    for c in "ABCD":
        rec[c]["label"] = label(rec[c]["answer"], q["answers"])
    return rec


def bootstrap_diff(recs, c1, c2, lab, B=2000, seed=0):
    x = [(r[c2]["label"] == lab) - (r[c1]["label"] == lab) for r in recs]
    n = len(x)
    rng = random.Random(seed)
    means = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return 100 * sum(x) / n, 100 * means[int(.025 * B)], 100 * means[int(.975 * B)]


def summarize(recs, out_dir, qstats, cmeta, args):
    n = len(recs)
    S = {"n": n, "model": args.model, "k": args.k, "question_stats": qstats, "corpus": cmeta, "conditions": {}}
    for c in "ABCD":
        L = Counter(r[c]["label"] for r in recs)
        answered = L["correct"] + L["incorrect"]
        S["conditions"][c] = {
            "correct": L["correct"], "incorrect": L["incorrect"], "abstain": L["abstain"],
            "correct_pct": 100 * L["correct"] / n, "incorrect_pct": 100 * L["incorrect"] / n,
            "abstain_pct": 100 * L["abstain"] / n,
            "selective_acc": 100 * L["correct"] / answered if answered else float("nan"),
            "calls_per_q": sum(r[c]["calls"] for r in recs) / n,
            "tokens_per_q": sum(r[c]["tok_in"] + r[c]["tok_out"] for r in recs) / n,
            "retrieval_ms_per_q": (sum(r["retrieval_ms"] for r in recs) / n) if c in "CD" else 0,
        }
    S["recall_at_k"] = 100 * sum(r["gold_in_topk"] for r in recs) / n
    # checker table
    T = defaultdict(Counter)
    for r in recs:
        row = r["C"]["label"]
        d = r["D"]["decision"]
        if d == "CORRECT":
            col = "corrected_right" if r["D"]["label"] == "correct" else "corrected_wrong"
        else:
            col = d.lower()
        T[row][col] += 1
    S["checker"] = {k: dict(v) for k, v in T.items()}
    S["checker_unparsed"] = sum(not r["D"]["parsed"] for r in recs)
    # errors by retrieval
    for c in "CD":
        inc = [r for r in recs if r[c]["label"] == "incorrect"]
        S[f"{c}_incorrect_gold_missing"] = sum(not r["gold_in_topk"] for r in inc)
    # A->B: abstentions on questions A had right
    S["B_abstain_where_A_correct"] = sum(r["B"]["label"] == "abstain" and r["A"]["label"] == "correct" for r in recs)
    S["D_withdrew_correct_gold_missing"] = sum(r["C"]["label"] == "correct" and r["D"]["label"] == "abstain"
                                               and not r["gold_in_topk"] for r in recs)
    S["ci"] = {}
    for a, b in [("A", "B"), ("B", "C"), ("C", "D")]:
        S["ci"][f"{a}->{b}"] = {lab: bootstrap_diff(recs, a, b, lab) for lab in ("correct", "incorrect")}
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    names = {"A": "A: Direct", "B": "B: + Abstain", "C": "C: + Retrieval", "D": "D: + Verify"}
    lines = [f"Model: {args.model}   n = {n}   k = {args.k}   Recall@{args.k} = {S['recall_at_k']:.1f}%", "",
             "| Condition | Correct % | Incorrect % | Abstain % | Selective acc. % | Calls/q | Tokens/q | Retrieval ms/q |",
             "|---|---|---|---|---|---|---|---|"]
    for c in "ABCD":
        s = S["conditions"][c]
        lines.append(f"| {names[c]} | {s['correct_pct']:.1f} | {s['incorrect_pct']:.1f} | {s['abstain_pct']:.1f} | "
                     f"{s['selective_acc']:.1f} | {s['calls_per_q']:.0f} | {s['tokens_per_q']:.0f} | "
                     f"{s['retrieval_ms_per_q']:.1f} |")
    lines += ["", "Checker decisions (rows = C's draft):", "",
              "| C draft | Keep | Corrected-right | Corrected-wrong | Withdraw |", "|---|---|---|---|---|"]
    for row in ("correct", "incorrect", "abstain"):
        t = S["checker"].get(row, {})
        lines.append(f"| {row} | {t.get('keep',0)} | {t.get('corrected_right',0)} | {t.get('corrected_wrong',0)} | {t.get('withdraw',0)} |")
    lines += ["", "Paired differences, percentage points (95% bootstrap CI):"]
    for k_, v in S["ci"].items():
        lines.append(f"  {k_}: correct {v['correct'][0]:+.1f} [{v['correct'][1]:+.1f}, {v['correct'][2]:+.1f}]  "
                     f"incorrect {v['incorrect'][0]:+.1f} [{v['incorrect'][1]:+.1f}, {v['incorrect'][2]:+.1f}]")
    lines += ["", f"B abstained on {S['B_abstain_where_A_correct']} questions that A answered correctly",
              f"C wrong answers with gold missing from top-k: {S['C_incorrect_gold_missing']} / {S['conditions']['C']['incorrect']}",
              f"D withdrew a correct C answer while gold was missing from passages: {S['D_withdrew_correct_gold_missing']}",
              f"Checker outputs that could not be parsed: {S['checker_unparsed']}"]
    open(os.path.join(out_dir, "tables.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # sample for manual labelling by both authors
    samp = random.Random(1).sample(recs, min(100, n))
    with open(os.path.join(out_dir, "annotation_sample.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question", "gold", "condition", "answer", "auto_label", "passage_ids",
                    "annotator1_correct", "annotator2_correct", "evidence(support/contradict/insufficient)"])
        for r in samp:
            for c in "ABCD":
                w.writerow([r["qid"], r["question"], " | ".join(r["answers"]), c, r[c]["answer"], r[c]["label"],
                            " ; ".join(r["passage_ids"]) if c in "CD" else "", "", "", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--pages-per-q", type=int, default=3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true", help="offline smoke test with fake model/corpus")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    qs, qstats = load_questions("data/NQ-open.dev.jsonl", args.n, args.seed)
    print(f"questions: {qstats}")
    passages, cmeta = build_corpus(qs, args.out, args.pages_per_q, fake=args.dry_run)
    print(f"corpus: {cmeta}")
    ret = Retriever(passages)
    llm = Dummy() if args.dry_run else Ollama(args.model, args.host)

    rec_path = os.path.join(args.out, "records.jsonl")
    done = {}
    if os.path.exists(rec_path):
        for l in open(rec_path):
            r = json.loads(l)
            done[r["qid"]] = r
    t0 = time.time()
    with open(rec_path, "a") as f:
        for i, q in enumerate(qs):
            if q["qid"] in done:
                continue
            r = run_one(q, llm, ret, args.k)
            done[q["qid"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(qs)}  ({el/60:.1f} min)")
    recs = [done[q["qid"]] for q in qs]
    summarize(recs, args.out, qstats, cmeta, args)


if __name__ == "__main__":
    main()
