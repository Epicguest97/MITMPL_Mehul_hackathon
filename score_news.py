"""
score_news.py — Score un-scored headlines with FinBERT.
First run downloads the model (~400MB). Usage: python score_news.py
"""
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "yiyanghkust/finbert-tone"
LABELS = ["positive", "negative", "neutral"]

def init_db():
    conn = sqlite3.connect("stock_nlp.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            news_id INTEGER PRIMARY KEY,
            pos REAL, neg REAL, neu REAL, score REAL,
            FOREIGN KEY (news_id) REFERENCES news(id)
        )""")
    return conn

def main():
    conn = init_db()
    rows = conn.execute("""
        SELECT n.id, n.title FROM news n
        LEFT JOIN scores s ON n.id = s.news_id
        WHERE s.news_id IS NULL
    """).fetchall()
    print(f"{len(rows)} headlines to score...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()

    BATCH = 32
    done = 0
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            texts = [t[:512] for _, t in batch]  # FinBERT max length
            inputs = tokenizer(texts, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            for (nid, _), p in zip(batch, probs):
                pos, neg, neu = p[0].item(), p[1].item(), p[2].item()
                score = pos - neg
                conn.execute(
                    "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)",
                    (nid, pos, neg, neu, score),
                )
            conn.commit()
            done += len(batch)
            print(f"  scored {done}/{len(rows)}")
    print("Done.")

if __name__ == "__main__":
    main()
