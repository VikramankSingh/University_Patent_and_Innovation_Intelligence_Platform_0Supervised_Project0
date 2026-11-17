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

st.set_page_config(page_title="Patent & Innovation Intelligence", layout="wide")

CLUSTER_NAMES = {
    0: "Disease Prediction Models",
    1: "Medical Image Segmentation",
    2: "Text & Question Answering",
    3: "Patient Care & Treatment",
    4: "Wireless Communication",
    5: "Human-Machine Interaction & Robotics",
    6: "Wearables & Physiological Sensing",
    7: "Cancer & Microbiome Biotech",
    8: "3D & Anatomical Imaging",
    9: "Core ML Training Methods",
    10: "Data Storage & Blockchain",
    11: "Deep Learning Imaging Networks",
}


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
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return clf, vec, encoder


@st.cache_resource
def load_explainer(_clf, _vec, _df):
    sample = _df[_df["domain"] != "Other"].sample(300, random_state=0)["abstract"]
    return shap.LinearExplainer(_clf, _vec.transform(sample))


df, pairs, emb = load_data()
clf, vec, encoder = load_models()
explainer = load_explainer(clf, vec, df)

st.title("University Patent & Innovation Intelligence Platform")
st.caption("Search, classify, cluster and map patents using NLP and machine learning.")

st.sidebar.header("Dataset")
st.sidebar.metric("Patents", f"{len(df):,}")
st.sidebar.metric("Assignees", f"{df['primary_assignee'].nunique():,}")
st.sidebar.metric("Years covered", f"{int(df['year'].min())}-{int(df['year'].max())}")
st.sidebar.metric("Overlap pairs found", f"{len(pairs):,}")

tab_search, tab_map, tab_trends, tab_net, tab_dup = st.tabs(
    ["Search & Classify", "Cluster Map", "Trends", "Technology Network", "Overlap Checker"]
)

# ---------------- Search & Classify ----------------
with tab_search:
    st.subheader("Describe an invention")
    query = st.text_area(
        "Paste an abstract or describe your idea",
        height=150,
        placeholder="e.g. A deep learning system that detects tumors in MRI scans",
    )
    top_k = st.slider("Number of similar patents", 3, 15, 5)

    if st.button("Analyze") and query.strip():
        col1, col2 = st.columns(2)

        X = vec.transform([query])
        pred = clf.predict(X)[0]
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

        with col1:
            st.success(f"Predicted domain: {pred}")
            if hasattr(clf, "predict_proba"):
                proba = pd.Series(clf.predict_proba(X)[0], index=clf.classes_)
                if proba.max() < 0.4:
                    st.caption("Low confidence: short descriptions are hard to classify. Paste a full abstract for better results.")
                proba = proba.sort_values(ascending=False).head(5)
                fig = px.bar(x=proba.values[::-1], y=proba.index[::-1], orientation="h",
                             labels={"x": "Probability", "y": ""})
                st.plotly_chart(fig)

        with col2:
            nearest_clusters = df.iloc[top[:10]]["cluster_label"].value_counts()
            st.info(f"Closest technology cluster: {nearest_clusters.index[0]}")
            if sims[top[0]] >= 0.90:
                st.warning(f"Possible overlap: very similar to an existing patent ({sims[top[0]]:.2f}).")
            else:
                st.write(f"Highest similarity to any existing patent: {sims[top[0]]:.2f}")

        st.subheader("Most similar patents")
        results = df.iloc[top][["title", "assignee", "year", "domain", "cluster_label"]].copy()
        results.insert(0, "similarity", sims[top].round(3))
        st.dataframe(results)

        for i in top[:3]:
            with st.expander(df.loc[i, "title"]):
                st.write(df.loc[i, "abstract"])

        st.subheader("Why this prediction? (SHAP)")
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
            st.write("No known words found in the text, so there is nothing to explain.")
        else:
            expl_df = pd.DataFrame({"word": words[present], "impact": vals[present]})
            expl_df = expl_df.reindex(expl_df["impact"].abs().sort_values(ascending=False).index).head(10)
            expl_df["effect"] = np.where(expl_df["impact"] >= 0, "pushes toward " + pred, "pushes away")
            fig = px.bar(expl_df[::-1], x="impact", y="word", color="effect", orientation="h",
                         labels={"impact": "SHAP value", "word": ""})
            st.plotly_chart(fig)
            st.caption("Each bar shows how much a word in your text pushed the model toward or away from the predicted domain.")

# ---------------- Cluster Map ----------------
with tab_map:
    st.subheader("Innovation map (each dot is a patent)")
    color_by = st.radio("Color by", ["cluster_label", "domain"], horizontal=True)
    sample = df.sample(min(len(df), 8000), random_state=1)
    fig = px.scatter(sample, x="umap_x", y="umap_y", color=color_by,
                     hover_data=["title", "assignee", "year"], opacity=0.7, height=700)
    fig.update_traces(marker=dict(size=4))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig)

# ---------------- Trends ----------------
with tab_trends:
    st.subheader("Patent activity over time")
    d = df.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)
    d = d[d["year"] >= 2010]

    group = st.radio("Group by", ["cluster_label", "domain"], horizontal=True, key="grp")
    yearly = d.groupby(["year", group]).size().reset_index(name="patents")
    st.plotly_chart(px.area(yearly, x="year", y="patents", color=group))
    st.caption("The drop in the latest year is likely due to publication lag (recent filings are not all published yet).")

    st.subheader("Emerging areas (change in share of patents)")
    early = d[(d["year"] >= 2018) & (d["year"] <= 2020)][group].value_counts(normalize=True)
    recent = d[(d["year"] >= 2022) & (d["year"] <= 2024)][group].value_counts(normalize=True)
    growth = pd.DataFrame({"share_2018_2020": early, "share_2022_2024": recent}).fillna(0)
    growth["change_pct_points"] = ((growth["share_2022_2024"] - growth["share_2018_2020"]) * 100).round(2)
    growth = growth.sort_values("change_pct_points", ascending=False)
    st.dataframe(growth.round(3))

    st.subheader("Academic vs industry")
    acad = d.groupby("year")["is_academic"].mean().reset_index()
    acad["academic_share_pct"] = acad["is_academic"] * 100
    st.plotly_chart(px.line(acad, x="year", y="academic_share_pct", markers=True))
    st.caption("Academic flag is keyword-based (university, institute, hospital...) and likely undercounts non-English names.")

# ---------------- Technology Network ----------------
with tab_net:
    st.subheader("How technology areas connect")
    ids = sorted(CLUSTER_NAMES)
    cent = np.vstack([emb[(df["cluster"] == c).values].mean(axis=0) for c in ids])
    cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    sim = cent @ cent.T
    off_diag = sim[~np.eye(len(ids), dtype=bool)]
    lo, hi = float(off_diag.min()), float(off_diag.max())
    thr = st.slider("Link threshold (higher = fewer links)", round(lo, 2), round(hi, 2),
                    round(float(np.percentile(off_diag, 70)), 2), 0.01)

    G = nx.Graph()
    for k, c in enumerate(ids):
        G.add_node(c, size=int((df["cluster"] == c).sum()))
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if sim[a, b] >= thr:
                G.add_edge(ids[a], ids[b], weight=float(sim[a, b]))

    pos = nx.spring_layout(G, seed=3, weight="weight")
    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="#888"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        text=[CLUSTER_NAMES[n] for n in G.nodes()], textposition="top center",
        marker=dict(size=[G.nodes[n]["size"] ** 0.5 / 2 for n in G.nodes()],
                    color="#4C78A8", opacity=0.8),
        hovertext=[f"{CLUSTER_NAMES[n]}: {G.nodes[n]['size']} patents" for n in G.nodes()],
        hoverinfo="text"))
    fig.update_layout(showlegend=False, height=650,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig)
    st.caption(f"{G.number_of_edges()} links between {G.number_of_nodes()} technology areas. "
               "Node size = number of patents.")

# ---------------- Overlap Checker ----------------
with tab_dup:
    st.subheader("Duplicate and overlapping patents")
    types = sorted(pairs["type"].unique())
    chosen = st.multiselect("Overlap type", types, default=types)
    cross_only = st.checkbox("Only pairs with different owners")
    view = pairs[pairs["type"].isin(chosen)]
    if cross_only:
        view = view[~view["same_assignee"]]
    st.write(f"{len(view):,} pairs")
    st.dataframe(view[["similarity", "type", "title_a", "assignee_a",
                       "title_b", "assignee_b", "same_assignee"]])