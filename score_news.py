"""
score_news.py — FinBERT-score un-scored articles. Reads/writes DuckDB directly.
Usage: python score_news.py [subset]      (default: all subsets)
"""
import sys
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from db import get_con

MODEL = "yiyanghkust/finbert-tone"

def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else None
    con = get_con()
    where = "AND n.subset = ?" if subset else ""
    params = [subset] if subset else []
    df = con.execute(f"""
        SELECT n.news_id, n.text
        FROM raw_news n LEFT JOIN scores s USING(news_id)
        WHERE s.news_id IS NULL {where}
        ORDER BY n.news_id
    """, params).df()
    print(f"{len(df)} articles to score...")
    if df.empty:
        return

    tok = BertTokenizer.from_pretrained(MODEL)
    model = BertForSequenceClassification.from_pretrained(MODEL)
    model.eval()

    B = 32
    out = []
    with torch.no_grad():
        for i in range(0, len(df), B):
            texts = [t[:512] for t in df["text"].iloc[i:i+B]]
            inputs = tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            for nid, p in zip(df["news_id"].iloc[i:i+B], probs):
                out.append((int(nid), p[1].item(), p[2].item(), p[0].item(),
                            p[1].item() - p[2].item()))
            if i and i % (B*200) == 0:
                print(f"  {i}/{len(df)}")
    res = pd.DataFrame(out, columns=["news_id", "pos", "neg", "neu", "score"])
    con.register("_s", res)
    con.execute("INSERT OR REPLACE INTO scores SELECT news_id,pos,neg,neu,score FROM _s")
    print(f"Scored {len(res)} articles.")

if __name__ == "__main__":
    main()
