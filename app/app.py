import html
import numpy as np
import pandas as pd
import joblib
import shap
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="Patent Atlas", page_icon="📐", layout="wide",
                   initial_sidebar_state="collapsed")

CLUSTER_NAMES = {
    0: "Disease Prediction Models", 1: "Medical Image Segmentation",
    2: "Text & Question Answering", 3: "Patient Care & Treatment",
    4: "Wireless Communication", 5: "Human-Machine Interaction & Robotics",
    6: "Wearables & Physiological Sensing", 7: "Cancer & Microbiome Biotech",
    8: "3D & Anatomical Imaging", 9: "Core ML Training Methods",
    10: "Data Storage & Blockchain", 11: "Deep Learning Imaging Networks",
}
DOMAIN_COLORS = {
    "Medical Imaging": "#5CC8FF", "Clinical Decision Support": "#F6D55C",
    "General ML & Data Methods": "#B9A5FF", "Patient Monitoring & Wearables": "#7EE0B5",
    "Health Records & Operations": "#FFB27A", "Drug Discovery & Genomics": "#FF8FA3",
    "Language & Knowledge Systems": "#9FD36B", "Networking & Hardware": "#8FB3FF",
    "Security & Privacy": "#E9A6E8", "Autonomous Systems & Mobility": "#D8C79A",
    "Other": "#8EA2BC",
}
PALETTE = list(DOMAIN_COLORS.values())
YELLOW, CYAN, INK, LINE = "#F6D55C", "#5CC8FF", "#0F2A4A", "#2B5486"

EXAMPLES = {
    "MRI tumor scan": "A convolutional neural network system for segmenting tumors in magnetic resonance images, wherein the network receives a three-dimensional MRI volume, identifies a region of interest and outputs a segmentation mask used for radiotherapy planning.",
    "Heart-rate wearable": "A wearable device comprising a plurality of sensors configured to continuously monitor heart rate, blood oxygen saturation and sleep patterns of a user, wherein a machine learning model analyzes the sensor data to detect irregular cardiac rhythms and transmits an alert to a mobile device.",
    "Private health records": "A blockchain based system for securely storing and sharing electronic health records, wherein patient data is encrypted and access is controlled by smart contracts and biometric authentication to protect privacy.",
    "Sepsis risk alert": "A system for predicting the risk of sepsis in hospitalized patients by analyzing vital signs, laboratory results and clinical notes with a gradient boosted model, and alerting the care team when the predicted risk exceeds a threshold.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, .stApp, .stMarkdown, p, label, li, button, textarea, input { font-family: 'Instrument Sans', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons { font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
.stApp {
  background-color: #0F2A4A;
  background-image: linear-gradient(rgba(92,200,255,.07) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(92,200,255,.07) 1px, transparent 1px);
  background-size: 32px 32px;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 2.2rem; }
h1, h2, h3, .hero h1 { font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; }
.hero h1 { font-size: clamp(2.4rem, 6vw, 4.4rem); line-height: 1.02; font-weight: 800; color: #F4F8FF; margin: 0 0 .7rem; max-width: 16ch; }
.hero p { color: #B7C8DF; font-size: 1.12rem; max-width: 58ch; line-height: 1.55; margin: 0; }
.facts { display: flex; flex-wrap: wrap; gap: .4rem 2.2rem; margin: 1.5rem 0 1.2rem; padding: .8rem 0; border-top: 1px solid #2B5486; border-bottom: 1px solid #2B5486; }
.facts div { color: #B7C8DF; font-size: .95rem; }
.facts b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.5rem; color: #F6D55C; font-weight: 700; margin-right: .35rem; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #2B5486; }
.stTabs [data-baseweb="tab"] { padding: .5rem 0; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; font-weight: 700; color: #8EA2BC; background: transparent; }
.stTabs [aria-selected="true"] { color: #F6D55C; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #F6D55C; height: 3px; }
.stTextArea textarea { background: rgba(8,26,48,.85); border: 1px solid #2B5486; border-radius: 4px; color: #F4F8FF; font-size: 1.02rem; line-height: 1.5; }
.stTextArea textarea:focus { border-color: #F6D55C; box-shadow: none; }
button[data-testid="stBaseButton-primary"], button[kind="primary"] { background: #F6D55C; color: #0F2A4A; border: 0; border-radius: 4px; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; padding: .55rem 1.6rem; }
button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover { background: #FFE68A; color: #0F2A4A; }
button[data-testid="stBaseButton-secondary"], button[kind="secondary"] { background: transparent; border: 1px dashed #5C86B8; border-radius: 999px; color: #B7C8DF; font-size: .88rem; }
button[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover { border-color: #5CC8FF; color: #5CC8FF; }
.verdict { display: grid; grid-template-columns: 1.3fr 1fr 1fr; border: 1px solid #2B5486; background: rgba(8,26,48,.8); border-radius: 4px; margin: 1.2rem 0; }
.verdict > div { padding: 1.1rem 1.3rem; border-right: 1px solid #2B5486; }
.verdict > div:last-child { border-right: 0; }
.verdict small { display: block; color: #8EA2BC; font-size: .85rem; margin-bottom: .35rem; }
.verdict .big { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 800; line-height: 1.1; }
.verdict .mid { font-size: 1.12rem; font-weight: 600; color: #F4F8FF; line-height: 1.3; }
.pcard { border-left: 3px solid var(--c); background: rgba(8,26,48,.7); padding: .85rem 1.1rem; margin-bottom: .6rem; border-radius: 0 4px 4px 0; }
.pcard .t { color: #F4F8FF; font-weight: 600; font-size: 1.03rem; margin: .35rem 0 .15rem; }
.pcard .m { color: #8EA2BC; font-size: .88rem; }
.pcard .m span { margin-right: 1.1rem; }
.pcard .top { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.pcard .s { color: #F6D55C; font-weight: 700; font-family: 'Bricolage Grotesque', sans-serif; }
.bar { height: 3px; background: #1B3F68; margin-top: .6rem; }
.bar i { display: block; height: 3px; background: var(--c); }
.chip { display: inline-block; border: 1px solid; border-radius: 999px; padding: .08rem .7rem; font-size: .82rem; font-weight: 600; }
.note { color: #8EA2BC; font-size: .9rem; }
div[data-testid="stExpander"] { border: 1px solid #2B5486; border-radius: 4px; background: rgba(8,26,48,.55); }
@media (max-width: 760px) { .verdict { grid-template-columns: 1fr; } .verdict > div { border-right: 0; border-bottom: 1px solid #2B5486; } }
</style>
"""


def chip(label):
    c = DOMAIN_COLORS.get(label, "#8EA2BC")
    return f'<span class="chip" style="color:{c};border-color:{c}">{html.escape(str(label))}</span>'


def style(fig, height=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", colorway=PALETTE,
                      font=dict(family="Instrument Sans, sans-serif", color="#D5E1F2"),
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(gridcolor="rgba(92,200,255,.12)", zerolinecolor=LINE)
    fig.update_yaxes(gridcolor="rgba(92,200,255,.12)", zerolinecolor=LINE)
    if height:
        fig.update_layout(height=height)
    return fig


@st.cache_resource
def load_data():
    df = pd.read_csv(PROC / "patents_final.csv")
    df["cluster_label"] = df["cluster"].map(CLUSTER_NAMES)
    pairs = pd.read_csv(PROC / "duplicate_pairs.csv")
    emb = np.load(PROC / "embeddings.npy")
    return df, pairs, emb


@st.cache_resource
def load_models():
    pipe = joblib.load(PROC / "classifier_pipeline.joblib")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return pipe.named_steps["clf"], pipe.named_steps["tfidf"], encoder


@st.cache_resource
def load_explainer(_clf, _vec, _df):
    sample = _df[_df["domain"] != "Other"].sample(300, random_state=0)["abstract"]
    return shap.LinearExplainer(_clf, _vec.transform(sample))


df, pairs, emb = load_data()
clf, vec, encoder = load_models()
explainer = load_explainer(clf, vec, df)


def set_query(text):
    st.session_state["query"] = text


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"""
<div class="hero">
  <h1>Has someone already invented this?</h1>
  <p>Describe an idea or paste an abstract. Patent Atlas compares it with {len(df):,} machine learning and healthcare patents, tells you where it fits and shows who is working nearby.</p>
</div>
<div class="facts">
  <div><b>{len(df):,}</b>patents indexed</div>
  <div><b>{df['primary_assignee'].nunique():,}</b>organisations</div>
  <div><b>{int(df['year'].min())} to {int(df['year'].max())}</b>filing years</div>
  <div><b>{len(pairs):,}</b>overlapping pairs found</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.write("University Patent & Innovation Intelligence Platform, a supervised machine learning course project.")
    st.write("Data: Machine Learning and Healthcare Patent Dataset (Kaggle, CC0), from Google Patents.")

tab_check, tab_map, tab_trends, tab_net, tab_dup = st.tabs(
    ["Check an idea", "Patent map", "Trends", "Connections", "Overlaps"])

# ---------------- Check an idea ----------------
with tab_check:
    query = st.text_area("Your idea or abstract", key="query", height=140,
                         placeholder="e.g. A deep learning system that detects tumors in MRI scans",
                         label_visibility="collapsed")
    st.markdown('<div class="note">Not sure what to write? Try one of these:</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, key=f"ex_{name}", on_click=set_query, args=(text,))
    top_k = st.slider("How many similar patents to show", 3, 15, 5)

    if st.button("Check my idea", type="primary") and query.strip():
        X = vec.transform([query])
        pred = clf.predict(X)[0]
        proba = pd.Series(clf.predict_proba(X)[0], index=clf.classes_) if hasattr(clf, "predict_proba") else None
        q = encoder.encode([query], normalize_embeddings=True)[0]
        sims = emb @ q
        order = np.argsort(-sims)[:top_k * 4]
        keep, seen = [], set()
        for i in order:
            t = str(df.loc[i, "title"]).strip().lower()
            if t not in seen:
                seen.add(t)
                keep.append(i)
            if len(keep) == top_k:
                break
        top = np.array(keep)

        near = df.iloc[top[:10]]["cluster_label"].value_counts().index[0]
        best = float(sims[top[0]])
        if best >= 0.90:
            novelty = f'<span style="color:#FF8FA3">Possible overlap</span><br><span class="note">Very close to an existing patent ({best:.0%})</span>'
        else:
            novelty = f'<span style="color:#7EE0B5">No close match</span><br><span class="note">Highest similarity is {best:.0%}</span>'
        pc = DOMAIN_COLORS.get(pred, "#F4F8FF")
        conf = f'<span class="note">{proba.max():.0%} confidence</span>' if proba is not None else ""
        st.markdown(f"""
<div class="verdict">
  <div><small>Likely domain</small><div class="big" style="color:{pc}">{html.escape(str(pred))}</div>{conf}</div>
  <div><small>Nearest technology area</small><div class="mid">{html.escape(str(near))}</div></div>
  <div><small>Novelty check</small><div class="mid">{novelty}</div></div>
</div>""", unsafe_allow_html=True)
        if proba is not None and proba.max() < 0.4:
            st.caption("Low confidence: short descriptions are hard to classify. Paste a full abstract for a firmer answer.")

        st.subheader("Closest existing patents")
        for rank, i in enumerate(top):
            r, s = df.loc[i], float(sims[i])
            c = DOMAIN_COLORS.get(r["domain"], "#8EA2BC")
            yr = int(r["year"]) if pd.notna(r["year"]) else "n/a"
            st.markdown(f"""
<div class="pcard" style="--c:{c}">
  <div class="top">{chip(r['domain'])}<span class="s">{s:.0%} similar</span></div>
  <div class="t">{html.escape(str(r['title']))}</div>
  <div class="m"><span>{html.escape(str(r['assignee']))}</span><span>Filed {yr}</span><span>{html.escape(str(r['cluster_label']))}</span></div>
  <div class="bar"><i style="width:{max(s, 0) * 100:.0f}%"></i></div>
</div>""", unsafe_allow_html=True)
            if rank < 3:
                with st.expander("Read the abstract"):
                    st.write(r["abstract"])

        st.subheader("Why this domain?")
        sv = explainer.shap_values(X)
        class_idx = list(clf.classes_).index(pred)
        if isinstance(sv, list):
            vals = np.asarray(sv[class_idx])[0]
        else:
            sv = np.asarray(sv)
            vals = sv[0, :, class_idx] if sv.ndim == 3 else sv[0]
        words = vec.get_feature_names_out()
        present = X.nonzero()[1]
        if len(present) == 0:
            st.write("None of the words in your text are in the model's vocabulary, so there is nothing to explain.")
        else:
            ex = pd.DataFrame({"word": words[present], "impact": vals[present]})
            ex = ex.reindex(ex["impact"].abs().sort_values(ascending=False).index).head(10)
            toward = f"pushes toward {pred}"
            ex["effect"] = np.where(ex["impact"] >= 0, toward, "pushes away")
            fig = px.bar(ex[::-1], x="impact", y="word", color="effect", orientation="h",
                         color_discrete_map={toward: YELLOW, "pushes away": "#6C86A8"},
                         labels={"impact": "SHAP value", "word": ""})
            st.plotly_chart(style(fig))
            st.caption("Each bar shows how strongly a word in your text pushed the model toward or away from the predicted domain.")

# ---------------- Patent map ----------------
with tab_map:
    st.subheader("Every patent, placed by meaning")
    st.caption("Patents that describe similar ideas sit close together. Hover a dot to see which patent it is.")
    color_by = st.radio("Color by", ["cluster_label", "domain"], horizontal=True,
                        format_func=lambda x: "Technology area" if x == "cluster_label" else "Predicted domain label")
    sample = df.sample(min(len(df), 8000), random_state=1)
    fig = px.scatter(sample, x="umap_x", y="umap_y", color=color_by,
                     hover_data=["title", "assignee", "year"], opacity=0.75)
    fig.update_traces(marker=dict(size=4))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(style(fig, 700))

# ---------------- Trends ----------------
with tab_trends:
    st.subheader("What is being filed, and when")
    d = df.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)
    d = d[d["year"] >= 2010]
    group = st.radio("Group by", ["cluster_label", "domain"], horizontal=True, key="grp",
                     format_func=lambda x: "Technology area" if x == "cluster_label" else "Domain label")
    yearly = d.groupby(["year", group]).size().reset_index(name="patents")
    st.plotly_chart(style(px.area(yearly, x="year", y="patents", color=group)))
    st.caption("The drop in the latest year is likely publication lag: recent filings are not all published yet.")

    st.subheader("Areas gaining ground")
    early = d[(d["year"] >= 2018) & (d["year"] <= 2020)][group].value_counts(normalize=True)
    recent = d[(d["year"] >= 2022) & (d["year"] <= 2024)][group].value_counts(normalize=True)
    growth = pd.DataFrame({"share_2018_2020": early, "share_2022_2024": recent}).fillna(0)
    growth["change_pct_points"] = ((growth["share_2022_2024"] - growth["share_2018_2020"]) * 100).round(2)
    st.dataframe(growth.sort_values("change_pct_points", ascending=False).round(3))

    st.subheader("Universities and industry")
    acad = d.groupby("year")["is_academic"].mean().reset_index()
    acad["academic_share_pct"] = acad["is_academic"] * 100
    fig = px.line(acad, x="year", y="academic_share_pct", markers=True)
    fig.update_traces(line_color=YELLOW)
    st.plotly_chart(style(fig))
    st.caption("Academic filers are detected from keywords such as university, institute and hospital, so non-English names are undercounted.")

# ---------------- Connections ----------------
with tab_net:
    st.subheader("How technology areas relate")
    ids = sorted(CLUSTER_NAMES)
    cent = np.vstack([emb[(df["cluster"] == c).values].mean(axis=0) for c in ids])
    cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    sim = cent @ cent.T
    off = sim[~np.eye(len(ids), dtype=bool)]
    lo, hi = float(off.min()), float(off.max())
    thr = st.slider("Show links stronger than", round(lo, 2), round(hi, 2),
                    round(float(np.percentile(off, 70)), 2), 0.01)
    G = nx.Graph()
    for c in ids:
        G.add_node(c, size=int((df["cluster"] == c).sum()))
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if sim[a, b] >= thr:
                G.add_edge(ids[a], ids[b], weight=float(sim[a, b]))
    pos = nx.spring_layout(G, seed=3, weight="weight")
    ex_, ey_ = [], []
    for u, v in G.edges():
        ex_ += [pos[u][0], pos[v][0], None]
        ey_ += [pos[u][1], pos[v][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex_, y=ey_, mode="lines", line=dict(width=1, color="#5C86B8"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode="markers+text", text=[CLUSTER_NAMES[n] for n in G.nodes()], textposition="top center",
        marker=dict(size=[G.nodes[n]["size"] ** 0.5 / 2 for n in G.nodes()], color=YELLOW,
                    opacity=0.9, line=dict(color="#0F2A4A", width=2)),
        hovertext=[f"{CLUSTER_NAMES[n]}: {G.nodes[n]['size']} patents" for n in G.nodes()], hoverinfo="text"))
    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(style(fig, 650))
    st.caption(f"{G.number_of_edges()} links between {G.number_of_nodes()} technology areas. Bigger circles mean more patents.")

# ---------------- Overlaps ----------------
with tab_dup:
    st.subheader("Patents that say almost the same thing")
    types = sorted(pairs["type"].unique())
    chosen = st.multiselect("Overlap type", types, default=types)
    cross_only = st.checkbox("Only pairs with different owners")
    view = pairs[pairs["type"].isin(chosen)]
    if cross_only:
        view = view[~view["same_assignee"]]
    st.write(f"{len(view):,} pairs")
    st.dataframe(view[["similarity", "type", "title_a", "assignee_a", "title_b", "assignee_b", "same_assignee"]])