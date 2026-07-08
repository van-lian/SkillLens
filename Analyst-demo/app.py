import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
import plotly.io as pio
import os

# Global Plotly template — transparent bg, dark readable text, clean grid
pio.templates["clean"] = pio.templates["plotly_white"]
pio.templates["clean"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1e293b", size=12),
    title=dict(font=dict(color="#1e293b", size=15), x=0),
    xaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickfont=dict(color="#1e293b"),
        titlefont=dict(color="#475569"),
        zerolinecolor="#e2e8f0",
    ),
    yaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickfont=dict(color="#1e293b"),
        titlefont=dict(color="#475569"),
        zerolinecolor="#e2e8f0",
    ),
    legend=dict(font=dict(color="#1e293b")),
    coloraxis=dict(colorbar=dict(tickfont=dict(color="#1e293b"))),
)
pio.templates.default = "clean"


#  Page config 
st.set_page_config(
    page_title="Skill Gap Analyzer – EDA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS 
st.markdown("""
<style>
.metric-card {
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
    border-left: 4px solid #3b82f6;
    background: rgba(59,130,246,0.06);
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.85rem; opacity: 0.65; margin-top: 2px; }
.section-header { font-size: 1.25rem; font-weight: 700; margin-bottom: 4px; }
.section-sub { font-size: 0.875rem; opacity: 0.65; margin-bottom: 18px; }

.insight-box {
    background: rgba(59,130,246,0.08);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: 0.9rem;
    color: inherit;
}
.insight-box b { color: #2563eb; }

.conclusion-box {
    background: rgba(34,197,94,0.08);
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.9rem;
    color: inherit;
}
.conclusion-box b { color: #16a34a; }

.warning-box {
    background: rgba(245,158,11,0.08);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.9rem;
    color: inherit;
}
.warning-box b { color: #d97706; }
</style>
""", unsafe_allow_html=True)


#  Helper: parse semicolon-separated skill columns 
def parse_skills(series):
    counter = Counter()
    for cell in series.dropna():
        for sk in str(cell).split(";"):
            sk = sk.strip().lower()
            if sk:
                counter[sk] += 1
    return counter


#  Load real dataset 
@st.cache_data
def load_data():
    # Support running from different working directories
    candidates = [
        "skill_mapping_output.csv",
        "dataset/cleaned_data/skill_mapping_output.csv",
        os.path.join(os.path.dirname(__file__), "skill_mapping_output.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.error(" Could not find `skill_mapping_output.csv`. Place it in the same folder as app.py.")
    st.stop()


@st.cache_data
def compute_features(df):
    all_skills_counter  = parse_skills(df["all_skills"])
    tech_counter        = parse_skills(df["skills_technical"])
    soft_counter        = parse_skills(df["skills_soft"])
    tool_counter        = parse_skills(df["skills_tool"])
    domain_counter      = parse_skills(df["skills_domain"])

    # Co-occurrence
    co = defaultdict(Counter)
    for cell in df["all_skills"].dropna():
        skills = [s.strip().lower() for s in str(cell).split(";") if s.strip()]
        for s1, s2 in combinations(skills, 2):
            co[s1][s2] += 1
            co[s2][s1] += 1

    pairs = []
    for s1, partners in co.items():
        for s2, cnt in partners.items():
            if s1 < s2:
                pairs.append({"skill_a": s1, "skill_b": s2, "count": cnt})
    co_df = pd.DataFrame(pairs).sort_values("count", ascending=False).reset_index(drop=True)

    return all_skills_counter, tech_counter, soft_counter, tool_counter, domain_counter, co_df


df = load_data()
all_skills_counter, tech_counter, soft_counter, tool_counter, domain_counter, co_df = compute_features(df)

BLUE    = "#3b82f6"
RED     = "#ef4444"
GREEN   = "#22c55e"
ORANGE  = "#f97316"
PURPLE  = "#8b5cf6"

CAT_COLORS = {
    "Technical": BLUE,
    "Soft":      RED,
    "Tools":     GREEN,
    "Domain":    ORANGE,
}

#  Sidebar 
with st.sidebar:
    st.markdown("##  Skill Gap Analyzer")
    st.markdown("**EDA Dashboard**")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        [
            "Overview",
            "Dataset Quality",
            "Q1 · Frequent Skills",
            "Q2 · Job Complexity",
            "Q3 · Skill Co-occurrence",
            "Conclusions",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Data: <b>skill_mapping_output.csv</b><br>"
        "Notebook: <code>04_Data_EDA</code> &amp; <code>05_Data_EDA_BusinessQuestion</code></small>",
        unsafe_allow_html=True,
    )


# 
# PAGE: OVERVIEW
# 
if page == "Overview":
    st.title(" Skill Gap Analyzer — EDA Dashboard")
    st.markdown(
        "Exploratory Data Analysis of technology job postings and their required skills.  "
        )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{len(df):,}", "Total Job Postings", BLUE),
        (f"{len(all_skills_counter):,}", "Unique Skills", GREEN),
        (f"{df['skill_count'].mean():.1f}", "Avg Skills / Posting", ORANGE),
        (f"{int(df['skill_count'].median())}", "Median Skills / Posting", PURPLE),
    ]
    for col, (val, label, color) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(
                f'<div class="metric-card" style="border-color:{color}">'
                f'<div class="metric-value" style="color:{color}">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("###  Dataset Schema")
    schema = pd.DataFrame({
        "Column": ["title", "cleaned_text", "skills_technical", "skills_soft", "skills_tool",
                   "skills_domain", "skills_certification", "all_skills", "skill_count"],
        "Type":   ["String", "String", "Semicolon-separated", "Semicolon-separated",
                   "Semicolon-separated", "Semicolon-separated", "Semicolon-separated",
                   "Semicolon-separated", "Integer"],
        "Description": [
            "Job title of the posting",
            "Cleaned and tokenised job description text",
            "Programming languages, frameworks, ML methods",
            "Soft skills: communication, leadership, etc.",
            "Software tools: Docker, Git, Jira, etc.",
            "Business domain knowledge areas",
            "Required certifications",
            "All skills combined across all categories",
            "Total number of extracted skills",
        ],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.markdown("###  Business Questions")
    for q, text in [
        ("Q1", "What skills are most frequently required in technology-related job postings?"),
        ("Q2", "How complex are technology job requirements based on the number of required skills?"),
        ("Q3", "Which technical skills most frequently appear together in job postings?"),
    ]:
        st.markdown(f"**{q}** — {text}")

    st.markdown("###  Sample Data")
    st.dataframe(
        df[["title", "skills_technical", "skills_soft", "skills_tool", "skill_count"]].head(10),
        use_container_width=True,
        hide_index=True,
    )


# 
# PAGE: DATASET QUALITY
# 
elif page == "Dataset Quality":
    st.title(" Dataset Quality Check")
    st.markdown("Mirrors notebooks **04_Data_EDA** sections: Missing values, Duplicates, Text-length distribution.")
    st.markdown("---")

    #  Missing values 
    st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Count and percentage of nulls per column</div>', unsafe_allow_html=True)

    missing = df.isnull().sum().rename("missing").to_frame()
    missing["pct (%)"] = (missing["missing"] / len(df) * 100).round(1)
    missing_nonzero = missing[missing["missing"] > 0].sort_values("missing", ascending=False)

    if missing_nonzero.empty:
        st.success(" No missing values detected in any column.")
    else:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.dataframe(missing_nonzero, use_container_width=True)
        with col_right:
            fig = px.bar(
                missing_nonzero.reset_index(),
                x="index", y="pct (%)",
                labels={"index": "Column", "pct (%)": "Missing (%)"},
                color_discrete_sequence=[ORANGE],
                title="Missing Value % by Column",
            )
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box"> <b>Insight:</b> Skill columns (<code>skills_soft</code>, '
        '<code>skills_certification</code>, <code>skills_tool</code>) have high missing rates — '
        'this is expected because not every job posting mentions all skill categories. '
        'The <code>all_skills</code> column consolidates everything; rows with NaN there '
        '(~13%) are postings where <b>no skills were extracted at all</b> and are excluded '
        'from frequency and co-occurrence analyses.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    #  Duplicates 
    st.markdown('<div class="section-header">Duplicate Check</div>', unsafe_allow_html=True)
    n_dup = df.duplicated(subset=["cleaned_text"]).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Duplicate cleaned_text", f"{n_dup:,}")
    c3.metric("Unique Rows", f"{len(df) - n_dup:,}")
    if n_dup == 0:
        st.success(" No duplicates found in cleaned_text — dataset is clean.")
    else:
        st.warning(f" {n_dup} duplicate `cleaned_text` entries found — should be dropped before modelling.")

    st.markdown("---")

    #  Skill count distribution 
    st.markdown('<div class="section-header">Skill Count Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How many skills are listed per job posting?</div>', unsafe_allow_html=True)

    fig = px.histogram(
        df, x="skill_count",
        nbins=int(df["skill_count"].max()),
        color_discrete_sequence=[BLUE],
        labels={"skill_count": "Number of Skills", "count": "Job Postings"},
        title="Distribution of Skill Count per Job Posting",
    )
    fig.add_vline(x=df["skill_count"].mean(), line_dash="dash", line_color=RED,
                  annotation_text=f"Mean={df['skill_count'].mean():.1f}")
    fig.add_vline(x=df["skill_count"].median(), line_dash="dot", line_color=GREEN,
                  annotation_text=f"Median={int(df['skill_count'].median())}")
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"> <b>Insight:</b> The distribution is heavily right-skewed. '
        f'Most postings list only <b>1–3 skills</b> (median = {int(df["skill_count"].median())}), '
        f'while a long tail of specialist roles lists 15–28. '
        f'This reflects real-world variance in how thoroughly companies document skill requirements.</div>',
        unsafe_allow_html=True,
    )

    #  cleaned_text length 
    st.markdown("---")
    st.markdown('<div class="section-header">cleaned_text Length</div>', unsafe_allow_html=True)
    df_len = df.copy()
    df_len["word_count"] = df_len["cleaned_text"].dropna().apply(lambda x: len(str(x).split()))
    df_len["char_count"] = df_len["cleaned_text"].dropna().apply(len)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_len, x="word_count", nbins=60,
                           color_discrete_sequence=[PURPLE],
                           labels={"word_count": "Word Count"},
                           title="Word Count Distribution (cleaned_text)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_len, x="char_count", nbins=60,
                           color_discrete_sequence=[ORANGE],
                           labels={"char_count": "Character Count"},
                           title="Character Count Distribution (cleaned_text)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box"> <b>Insight:</b> Text length varies widely, '
        'from very short role summaries to detailed multi-paragraph descriptions. '
        'Short texts may reduce skill extraction recall — a consideration for the NLP pipeline.</div>',
        unsafe_allow_html=True,
    )


# 
# PAGE: Q1 – FREQUENT SKILLS
# 
elif page == "Q1 · Frequent Skills":
    st.title(" Q1 — Most Frequently Required Skills")
    st.markdown("*What skills appear most in technology job postings?*")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Top N Overall", "By Category", "Category Share"])

    with tab1:
        n_top = st.slider("Number of top skills to show", 10, 40, 20)
        top_n = pd.DataFrame(all_skills_counter.most_common(n_top), columns=["skill", "count"])

        fig = px.bar(top_n, x="count", y="skill", orientation="h",
                     color="count", color_continuous_scale="Blues",
                     labels={"count": "Job Postings", "skill": "Skill"},
                     title=f"Top {n_top} Most Frequently Required Skills")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False,
                          height=max(400, n_top * 28))
        st.plotly_chart(fig, use_container_width=True)

        top3 = [sk for sk, _ in all_skills_counter.most_common(3)]
        st.markdown(
            f'<div class="insight-box"> <b>Insight:</b> The top demanded skills are '
            f'<b>{top3[0].title()}</b>, <b>{top3[1].title()}</b>, and <b>{top3[2].title()}</b>. '
            f'Skill demand follows a <b>power-law distribution</b>: a small set of ~15 core skills '
            f'covers the vast majority of job requirements, while the remaining {len(all_skills_counter) - 15:,} '
            f'unique skills are niche or role-specific.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        categories = {
            "Technical": tech_counter,
            "Soft":      soft_counter,
            "Tools":     tool_counter,
            "Domain":    domain_counter,
        }
        top_k = st.slider("Top K per category", 5, 15, 10)
        cols = st.columns(2)

        for idx, (cat, counter) in enumerate(categories.items()):
            if not counter:
                continue
            top = pd.DataFrame(counter.most_common(top_k), columns=["skill", "count"])
            color = CAT_COLORS[cat]

            fig = px.bar(top, x="count", y="skill", orientation="h",
                         color_discrete_sequence=[color],
                         labels={"count": "Frequency", "skill": ""},
                         title=f"Top {top_k} — {cat} Skills")
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=350, margin=dict(l=10, r=10, t=40, b=30),
            )
            with cols[idx % 2]:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box"> <b>Insight:</b> '
            'Technical skills (Java, Python, Angular) dominate in raw volume. '
            'In the soft-skill category, <b>English</b> and <b>Leadership</b> top the list — '
            'reflecting that many of these postings are international roles where language '
            'proficiency is explicitly stated. Tools like <b>Git</b> and <b>Docker</b> '
            'are the most universally required utilities.</div>',
            unsafe_allow_html=True,
        )

    with tab3:
        cat_totals = {
            "Technical": sum(tech_counter.values()),
            "Soft":      sum(soft_counter.values()),
            "Tools":     sum(tool_counter.values()),
            "Domain":    sum(domain_counter.values()),
        }
        fig = go.Figure(data=[go.Pie(
            labels=list(cat_totals.keys()),
            values=list(cat_totals.values()),
            hole=0.45,
            marker_colors=[BLUE, RED, GREEN, ORANGE],
            textinfo="label+percent",
        )])
        fig.update_layout(title="Skill Mentions by Category")
        st.plotly_chart(fig, use_container_width=True)

        cat_df = pd.DataFrame({"Category": cat_totals.keys(), "Total Mentions": cat_totals.values()})
        cat_df["Share (%)"] = (cat_df["Total Mentions"] / cat_df["Total Mentions"].sum() * 100).round(1)
        st.dataframe(cat_df.sort_values("Total Mentions", ascending=False),
                     use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="insight-box"> <b>Insight:</b> Technical skills account for the largest '
            'share of all skill mentions — confirming that hard skills remain the primary hiring '
            'criterion. Domain knowledge is also substantial, reflecting that employers increasingly '
            'seek specialists with industry context, not just coding ability.</div>',
            unsafe_allow_html=True,
        )


# 
# PAGE: Q2 – JOB COMPLEXITY
# 
elif page == "Q2 · Job Complexity":
    st.title(" Q2 — Job Complexity by Skill Count")
    st.markdown("*How many skills does a typical tech job posting require?*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{df['skill_count'].mean():.1f}")
    c2.metric("Median", f"{int(df['skill_count'].median())}")
    c3.metric("Max", f"{df['skill_count'].max()}")
    c4.metric("Postings with ≥ 5 skills",
              f"{(df['skill_count'] >= 5).mean()*100:.0f}%")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Histogram", "Box Plot by Title", "CDF"])

    with tab1:
        fig = px.histogram(df, x="skill_count",
                           nbins=int(df["skill_count"].max()),
                           color_discrete_sequence=[BLUE],
                           labels={"skill_count": "Number of Skills Required",
                                   "count": "Number of Job Postings"},
                           title="Distribution of Skill Count per Job Posting")
        fig.add_vline(x=df["skill_count"].mean(), line_dash="dash", line_color=RED,
                      annotation_text=f"Mean={df['skill_count'].mean():.1f}")
        fig.add_vline(x=df["skill_count"].median(), line_dash="dot", line_color=GREEN,
                      annotation_text=f"Median={int(df['skill_count'].median())}")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div class="insight-box"> <b>Insight:</b> The distribution is strongly right-skewed '
            f'with a median of <b>{int(df["skill_count"].median())} skills</b>. '
            f'Most postings are lean on explicitly stated skills, but a meaningful tail of complex '
            f'roles requires 10–28 skills — typically senior engineering or architecture positions.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        # Top 10 most frequent titles for a meaningful box plot
        top_titles = df["title"].value_counts().head(10).index.tolist()
        df_top = df[df["title"].isin(top_titles)].copy()

        fig = px.box(
            df_top, x="skill_count", y="title",
            color="title",
            orientation="h",
            labels={"skill_count": "Number of Skills", "title": "Job Title"},
            title="Skill Count Distribution — Top 10 Job Titles",
        )
        fig.update_layout(showlegend=False, height=450,
                          yaxis={"categoryorder": "median ascending"})
        st.plotly_chart(fig, use_container_width=True)

        desc = df_top.groupby("title")["skill_count"].describe().round(2)
        st.dataframe(desc, use_container_width=True)

        st.markdown(
            '<div class="insight-box"> <b>Insight:</b> Skill count varies significantly across '
            'job titles. Engineering roles with broader stacks (e.g. Big Data, Full Stack) tend '
            'to list more skills than focused roles. This informs role-specific benchmarks in '
            'the Skill Gap Scoring Engine.</div>',
            unsafe_allow_html=True,
        )

    with tab3:
        sorted_counts = np.sort(df["skill_count"])
        cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_counts, y=cdf,
                                 mode="lines", line=dict(color=BLUE, width=2.5),
                                 name="CDF"))
        for threshold, color, label in [(3, "gray", "≤3"), (5, "orange", "≤5"), (10, "red", "≤10")]:
            pct = (df["skill_count"] <= threshold).mean() * 100
            fig.add_vline(x=threshold, line_dash="dot", line_color=color,
                          annotation_text=f"{pct:.0f}% {label}")

        fig.update_layout(
            title="CDF — Cumulative % of Jobs by Skill Count Threshold",
            xaxis_title="Number of Skills Required",
            yaxis_title="Cumulative % of Job Postings",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box"> <b>Insight:</b> The CDF shows that the majority of '
            'postings require 5 or fewer skills. A candidate covering the top 10 skills in '
            'their domain is likely competitive for nearly all postings — a key design input '
            'for the Skill Gap Scoring Engine.</div>',
            unsafe_allow_html=True,
        )


# 
# PAGE: Q3 – CO-OCCURRENCE
# 
elif page == "Q3 · Skill Co-occurrence":
    st.title(" Q3 — Skill Co-occurrence")
    st.markdown("*Which skills most frequently appear together in job postings?*")
    st.markdown("---")

    if co_df.empty:
        st.warning("Not enough data to compute co-occurrences. Check that `all_skills` column has valid entries.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Top Pairs", "Heatmap", "Network Graph"])

    with tab1:
        n_pairs = st.slider("Top N pairs to show", 10, 30, 20)
        top_pairs = co_df.head(n_pairs).copy()
        top_pairs["pair"] = top_pairs["skill_a"] + "  +  " + top_pairs["skill_b"]

        fig = px.bar(top_pairs, x="count", y="pair", orientation="h",
                     color="count", color_continuous_scale="Oranges",
                     labels={"count": "Co-occurrence Count", "pair": "Skill Pair"},
                     title=f"Top {n_pairs} Most Frequently Co-occurring Skill Pairs")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False,
                          height=max(400, n_pairs * 30))
        st.plotly_chart(fig, use_container_width=True)

        top_pair = top_pairs.iloc[0]
        st.markdown(
            f'<div class="insight-box"> <b>Insight:</b> The strongest co-occurrence is '
            f'<b>{top_pair["skill_a"].title()} + {top_pair["skill_b"].title()}</b> '
            f'({int(top_pair["count"])} postings). These represent the minimum viable skill '
            f'bundles for common tech roles and directly inform the role-to-skill mapping '
            f'in the Scoring Engine.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        top_n_hm = st.slider("Top N skills for heatmap", 8, 20, 12)
        top_skills_hm = [sk for sk, _ in all_skills_counter.most_common(top_n_hm)]
        matrix = pd.DataFrame(0, index=top_skills_hm, columns=top_skills_hm)

        for _, row in co_df.iterrows():
            a, b, cnt = row["skill_a"], row["skill_b"], row["count"]
            if a in top_skills_hm and b in top_skills_hm:
                matrix.loc[a, b] = cnt
                matrix.loc[b, a] = cnt

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="YlOrRd",
            text=matrix.values,
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(
            title=f"Co-occurrence Heatmap — Top {top_n_hm} Skills",
            height=520,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box"> <b>Insight:</b> The heatmap reveals skill clusters. '
            'Backend languages (Java, Node.js, Python) form a dense cluster, while mobile '
            'skills (Android, Kotlin) cluster separately. '
            'Cross-cluster links highlight full-stack and DevOps profiles.</div>',
            unsafe_allow_html=True,
        )

    with tab3:
        try:
            import networkx as nx

            n_net = st.slider("Top N pairs for network", 20, 60, 30)
            top_net = co_df.head(n_net)

            G = nx.Graph()
            for _, row in top_net.iterrows():
                G.add_edge(row["skill_a"], row["skill_b"], weight=row["count"])

            pos = nx.spring_layout(G, seed=42, k=1.2)

            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_sizes = [all_skills_counter.get(n, 1) * 0.3 + 15 for n in G.nodes()]
            node_labels = list(G.nodes())

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                     line=dict(width=0.8, color="#cbd5e1"),
                                     hoverinfo="none"))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                     text=node_labels, textposition="top center",
                                     textfont=dict(size=9),
                                     marker=dict(size=node_sizes, color=BLUE,
                                                 line=dict(width=1, color="white")),
                                     hovertext=node_labels, hoverinfo="text"))
            fig.update_layout(
                title=f"Skill Co-occurrence Network (top {n_net} pairs)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.warning("networkx not installed — run `pip install networkx`")

        st.markdown(
            '<div class="insight-box"> <b>Insight:</b> Central nodes (highly connected) '
            'represent <b>anchor skills</b> that bridge multiple role types — such as Java, '
            'Python, and Git. Peripheral nodes indicate niche or specialised skills tied '
            'to specific domains (e.g. Maya → game/3D roles).</div>',
            unsafe_allow_html=True,
        )


# 
# PAGE: CONCLUSIONS
# 
elif page == "Conclusions":
    st.title(" EDA Conclusions & Project Direction")
    st.markdown("---")

    st.markdown("###  Key Findings")

    # Compute dynamic values for conclusions
    top3_overall = [sk.title() for sk, _ in all_skills_counter.most_common(3)]
    top3_soft    = [sk.title() for sk, _ in soft_counter.most_common(3)]
    top3_tools   = [sk.title() for sk, _ in tool_counter.most_common(3)]
    median_skills = int(df["skill_count"].median())
    pct_ge5 = (df["skill_count"] >= 5).mean() * 100
    top_pair = co_df.iloc[0] if not co_df.empty else None

    findings = [
        (
            "Q1 — Skill Demand is Concentrated",
            BLUE,
            f"""
- **{top3_overall[0]}, {top3_overall[1]}, and {top3_overall[2]}** dominate across all job categories — appearing in the largest share of tech postings in the dataset.
- **{top3_soft[0]}** and **{top3_soft[1]}** are the most demanded soft skills, confirming that modern employers expect hybrid technical-interpersonal profiles.
- **Skill distribution follows a power law**: a small set of ~15 core skills covers the vast majority of job requirements across {len(all_skills_counter):,} unique skills extracted.
- Domain knowledge (**{", ".join([sk.title() for sk, _ in domain_counter.most_common(3)])}**) features prominently, showing employers want role-specific industry context.
""",
        ),
        (
            "Q2 — Job Complexity is Low-to-Moderate but Variable",
            ORANGE,
            f"""
- The **median posting requires only {median_skills} skills**, reflecting that many postings describe jobs in shorthand rather than exhaustive detail.
- Distribution is **strongly right-skewed**: senior and specialist roles (e.g. Big Data Engineer, Full Stack) demand 10–28 skills while many roles list only 1–3.
- **~{100 - pct_ge5:.0f}%** of postings list fewer than 5 skills — a key design input: the Skill Gap Scoring Engine must handle sparse skill data gracefully.
- **Outliers are real**: very long skill lists reflect genuinely complex, multi-disciplinary roles (ML + DevOps, mobile + backend).
""",
        ),
        (
            "Q3 — Skills Cluster into Technology Bundles",
            GREEN,
            f"""
{f'- The strongest co-occurrence is **{top_pair["skill_a"].title()} + {top_pair["skill_b"].title()}** — appearing together in **{int(top_pair["count"])} postings**.' if top_pair is not None else ""}
- Backend languages (Java, Node.js, Python) form a dense cluster; mobile skills (Android, Kotlin) form a separate one.
- **Tools like Git, Docker, and Jira** act as cross-cluster bridges, appearing alongside both frontend and backend skills.
- These co-occurrence patterns directly inform **role-to-skill mapping** in the Scoring Engine: a competitive candidate needs technical depth **plus** the tools canonical to their stack.
""",
        ),
    ]

    for title, color, body in findings:
        with st.expander(title, expanded=True):
            st.markdown(body)

    st.markdown("---")
    st.markdown("###  Data Quality Notes")
    n_dup = df.duplicated(subset=["cleaned_text"]).sum()
    pct_no_skills = df["all_skills"].isna().mean() * 100
    st.markdown(
        f'<div class="warning-box">'
        f' <b>High missing rates</b> in skill columns (<code>skills_soft</code>: '
        f'{df["skills_soft"].isna().mean()*100:.0f}%, <code>skills_certification</code>: '
        f'{df["skills_certification"].isna().mean()*100:.0f}%) — expected but reduces extraction coverage.<br>'
        f' <b>{pct_no_skills:.1f}%</b> of postings have no skills extracted at all in <code>all_skills</code> '
        f'— these are excluded from frequency and co-occurrence analyses.<br>'
        f' <b>{n_dup} duplicate</b> <code>cleaned_text</code> entries were identified and should be '
        f'dropped before modelling.<br>'
        f' <b>Skill extraction quality</b> depends on the underlying NLP/regex pipeline (SkillMapper). '
        f'Improving recall directly improves model performance.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("###  Notebook Reference")
    notebooks = pd.DataFrame({
        "Notebook": ["01.1 & 01.2", "02", "03", "04", "05 (BQ)"],
        "Focus": [
            "Data Filtering",
            "Data Merging",
            "Data Cleaning",
            "General EDA (text quality, missing values, duplicates)",
            "EDA — Business Questions (Q1–Q3)",
        ],
        "Key Output": [
            "Filtered raw postings",
            "Unified dataset",
            "cleaned_data.csv",
            "Word/char distributions, quality checks",
            "Skill frequency, complexity, co-occurrence",
        ],
    })
    st.dataframe(notebooks, use_container_width=True, hide_index=True)
