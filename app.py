import json
from datetime import datetime
import pandas as pd
import streamlit as st
from transformers import pipeline

DATA_PATH = "data.json"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_data
def load_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def get_sentiment_pipe():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",   
        device=-1
    )

def month_label_to_range(label: str):
    dt = datetime.strptime(label, "%b %Y")
    start = dt.replace(day=1)
    if dt.month == 12:
        end = dt.replace(year=dt.year + 1, month=1, day=1)
    else:
        end = dt.replace(month=dt.month + 1, day=1)
    return start, end


def main():
    st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")
    st.title("Brand Reputation Monitor (2023)")
    st.caption("Scraped from web-scraping.dev • Sentiment: Hugging Face Transformers")

    data = load_data(DATA_PATH)

    section = st.sidebar.radio("Navigate", ["Products", "Testimonials", "Reviews"])
    #produkti
    if section == "Products":
        st.header("Products")

        products = data["products"]

        for p in products:
            st.markdown("---")

            #slika
            if p.get("image_url"):
                st.image(p["image_url"], width=250)

            #ime
            st.subheader(p.get("title", "Unnamed product"))

            #opis
            if p.get("description"):
                st.write(p["description"])

            #cena
            if p.get("price") is not None:
                st.write(f"💰 **Price:** ${p['price']:.2f}")
            else:
                st.write("💰 **Price:** N/A")
    #testimonials
    elif section == "Testimonials":
        st.subheader("Testimonials")

        testimonials = data.get("testimonials", [])

        for t in testimonials:
            #username
            st.markdown(f"### 👤 {t.get('username', 'Anonymous')}")

            #description
            st.write(t.get("text", ""))

            #rating
            rating = t.get("rating", 0)
            st.write("⭐" * rating)

            st.divider()
    #reviews
    else:
        st.subheader("Reviews (2023) + Sentiment Analysis")

        df = pd.DataFrame(data.get("reviews", []))
        if df.empty:
            st.warning("No reviews found in data.json")
            return

        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date_dt"])

        df_2023 = df[df["date_dt"].dt.year == 2023].copy()
        if df_2023.empty:
            st.warning("No 2023 reviews found. Check your scraper output.")
            return

        months = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
        month_labels = [m.strftime("%b %Y") for m in months]

        selected = st.select_slider("Select month (2023)", options=month_labels, value="Mar 2023")
        start, end = month_label_to_range(selected)

        filtered = df_2023[(df_2023["date_dt"] >= start) & (df_2023["date_dt"] < end)].copy()

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            st.metric("Reviews in month", len(filtered))
        with colB:
            st.metric("All 2023 reviews", len(df_2023))
        with colC:
            st.write(f"Showing reviews from **{start.strftime('%Y-%m-%d')}** to **{(end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}**")

        if filtered.empty:
            st.info("No reviews in this month.")
            return

        #sentiment
        clf = get_sentiment_pipe()

        texts = filtered["text"].astype(str).tolist()
        preds = clf(texts, batch_size=16, truncation=True)

        filtered["sentiment"] = [p["label"] for p in preds]
        filtered["confidence"] = [float(p["score"]) for p in preds]

        #positive/negative 
        filtered["sentiment"] = filtered["sentiment"].map({"POSITIVE": "Positive", "NEGATIVE": "Negative"}).fillna(filtered["sentiment"])

        st.markdown("### Filtered reviews + predictions")
        show_cols = ["date", "rid", "rating", "text", "sentiment", "confidence"]
        st.dataframe(
            filtered.sort_values("date_dt", ascending=False)[show_cols],
            use_container_width=True
        )

        counts = filtered["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")

        overall_avg = filtered["confidence"].mean()
        by_label = filtered.groupby("sentiment")["confidence"].mean().reset_index().rename(columns={"confidence": "avg_confidence"})

        st.markdown("### Sentiment summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg confidence (overall)", f"{overall_avg:.3f}")
        with c2:
            pos_avg = by_label.loc[by_label["sentiment"] == "Positive", "avg_confidence"]
            st.metric("Avg confidence (Positive)", f"{pos_avg.iloc[0]:.3f}" if len(pos_avg) else "—")
        with c3:
            neg_avg = by_label.loc[by_label["sentiment"] == "Negative", "avg_confidence"]
            st.metric("Avg confidence (Negative)", f"{neg_avg.iloc[0]:.3f}" if len(neg_avg) else "—")

        st.bar_chart(counts.set_index("sentiment")["count"])

        st.caption("Avg confidence per sentiment (for this month):")
        st.dataframe(by_label, use_container_width=True)


if __name__ == "__main__":
    main()
