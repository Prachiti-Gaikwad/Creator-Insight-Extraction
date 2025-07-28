import streamlit as st
import pandas as pd
import together
import re
import json

# ------------------- Settings & API Key -------------------
st.set_page_config(page_title="Creator Insight Extraction", layout="wide")
st.sidebar.title("Settings")

# Set Together API key directly from secrets
together.api_key = st.secrets.get("TOGETHER_API_KEY", None)

# LLM toggle and ranking weights
use_llm = st.sidebar.checkbox("Use LLM to parse query", value=False)
st.sidebar.markdown("---")
st.sidebar.subheader("Ranking Weights")
weight_engagement = st.sidebar.slider("Engagement Rate (%)", 0.0, 1.0, 0.5, 0.05)
weight_followers = st.sidebar.slider("Follower Count", 0.0, 1.0, 0.3, 0.05)
weight_likes = st.sidebar.slider("Avg Likes/Comments", 0.0, 1.0, 0.2, 0.05)

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Mock_Creator_Engagement_Data.csv")
    # Normalize column names: lowercase, underscores, no leading/trailing spaces
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

df = load_data()
# st.write("Columns in data:", df.columns.tolist())  # Debug output hidden
categories = df['category'].unique()

# ------------------- Query Examples & History -------------------
st.title("Creator Insight Extraction (LLM via Together API)")

example_queries = [
    "Show me the top fashion creators with >10000 followers",
    "List wellness creators with high engagement",
    "Find tech creators for my campaign",
    "Top creators in beauty with >5000 followers"
]

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

with st.expander("Example Queries", expanded=False):
    for ex in example_queries:
        if st.button(f"Use: {ex}", key=ex):
            st.session_state['query'] = ex

# Query input
query = st.text_input("Enter your query:", st.session_state.get('query', example_queries[0]))
if query and (not st.session_state.query_history or st.session_state.query_history[-1] != query):
    st.session_state.query_history.append(query)

with st.expander("Query History", expanded=False):
    for q in st.session_state.query_history[-5:][::-1]:
        st.write(q)

# ------------------- Query Parsing -------------------
def extract_category_basic(query, categories):
    query_lower = query.lower()
    for category in categories:
        if category.lower() in query_lower:
            return category
    return None

def parse_query_llm(query):
    if not together.api_key:
        st.warning("Together API key not found; using basic extraction.")
        return None
    prompt = f"""
    Extract structured filters from this query: \"{query}\".
    Return JSON with fields: category, follower_filter (>,<,= value), sort_by.
    Example:
    Input: \"Show top fashion creators with >10000 followers\"
    Output: {{\"category\":\"fashion\",\"follower_filter\":\">10000\",\"sort_by\":\"engagement rate\"}}
    """
    try:
        response = together.Complete.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            prompt=prompt,
            max_tokens=100,
            temperature=0
        )
        # Debug output removed for production
        if 'choices' not in response or not response['choices']:
            st.error(f"LLM API error: {response}")
            return None
        parsed_text = response['choices'][0]['text'].strip()
        # Use regex to find the first {...} block in the text
        json_match = re.search(r'\{.*?\}', parsed_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed = json.loads(json_str.replace("'", '"'))
            except Exception:
                parsed = eval(json_str)
            return parsed
        else:
            st.error(f"LLM did not return JSON: {parsed_text}")
            return None
    except Exception as e:
        st.error(f"Parsing error: {e}")
        return None

# ------------------- Ranking Logic -------------------
def rank_creators(df, category, follower_filter=None, w_eng=0.5, w_fol=0.3, w_likes=0.2):
    df_filtered = df[df['category'].str.lower() == category.lower()].copy()
    # Parse follower filter (e.g., ">10000")
    if follower_filter:
        match = re.match(r'([><=])\s*(\d+)', follower_filter)
        if match:
            op, value = match.groups()
            value = int(value)
            if op == '>':
                df_filtered = df_filtered[df_filtered['follower_count'] > value]
            elif op == '<':
                df_filtered = df_filtered[df_filtered['follower_count'] < value]
            elif op == '=':
                df_filtered = df_filtered[df_filtered['follower_count'] == value]
    # Normalize follower score
    max_followers = df_filtered['follower_count'].max()
    if pd.notna(max_followers) and max_followers > 0:
        df_filtered['Follower_Score'] = df_filtered['follower_count'] / max_followers
    else:
        df_filtered['Follower_Score'] = 0
    # Ensure numeric types
    df_filtered['engagement_rate_(%)'] = pd.to_numeric(df_filtered['engagement_rate_(%)'], errors='coerce').fillna(0)
    df_filtered['average_likes/post'] = pd.to_numeric(df_filtered['average_likes/post'], errors='coerce').fillna(0)
    df_filtered['average_comments/post'] = pd.to_numeric(df_filtered['average_comments/post'], errors='coerce').fillna(0)
    # Combine likes and comments per post
    df_filtered['avg_likes_comments'] = df_filtered['average_likes/post'] + df_filtered['average_comments/post']
    # Compute score
    df_filtered['Score'] = (
        df_filtered['engagement_rate_(%)'] * w_eng
        + df_filtered['Follower_Score'] * w_fol
        + df_filtered['avg_likes_comments'] * w_likes
    )
    ranked = df_filtered.sort_values(by='Score', ascending=False)
    return ranked

# ------------------- Main App Logic -------------------
if query:
    # Parse query
    if use_llm:
        parsed = parse_query_llm(query)
        if parsed:
            st.success("LLM successfully extracted the following filters:")

            # Natural language summary with icons and color
            st.markdown(
                f"""
                <div style='background-color:#232946; padding:16px; border-radius:10px; color:#eebbc3; font-size:1.1em;'>
                    <b>🔎 For your query, the AI found:</b><br>
                    <span style='color:#eebbc3;'>🏷️ <b>Category:</b> <span style='color:#fffffe;'>{parsed.get('category', '—')}</span></span> &nbsp; | &nbsp;
                    <span style='color:#eebbc3;'>👥 <b>Follower Filter:</b> <span style='color:#fffffe;'>{parsed.get('follower_filter', '—')}</span></span> &nbsp; | &nbsp;
                    <span style='color:#eebbc3;'>🔽 <b>Sort By:</b> <span style='color:#fffffe;'>{parsed.get('sort_by', '—')}</span></span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Table with icons and color
            st.markdown(
                """
                <style>
                .llm-table td, .llm-table th {padding: 8px 16px;}
                .llm-table th {background: #232946; color: #eebbc3;}
                .llm-table td {background: #121629; color: #fffffe;}
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <table class="llm-table">
                    <tr>
                        <th>🏷️ Category</th>
                        <th>👥 Follower Filter</th>
                        <th>🔽 Sort By</th>
                    </tr>
                    <tr>
                        <td>{parsed.get('category', '—')}</td>
                        <td>{parsed.get('follower_filter', '—')}</td>
                        <td>{parsed.get('sort_by', '—')}</td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Show raw extracted JSON"):
                st.json(parsed)
            # Allow user to edit extracted filters
            category = st.selectbox("Category", categories, index=list(categories).index(parsed.get("category", categories[0])) if parsed.get("category") in categories else 0)
            follower_filter = st.text_input("Follower Filter (e.g. >10000)", parsed.get("follower_filter", ""))
        else:
            st.warning("LLM parsing failed; using basic extraction.")
            category = st.selectbox("Category", categories)
            follower_filter = st.text_input("Follower Filter (e.g. >10000)", "")
    else:
        category = st.selectbox("Category", categories, index=0)
        follower_filter = st.text_input("Follower Filter (e.g. >10000)", "")
        # Optionally, try to extract category from query
        cat_basic = extract_category_basic(query, categories)
        if cat_basic:
            st.info(f"Detected category: {cat_basic}")
            category = cat_basic
    # Show available categories and follower count range for debugging
    st.write("Available categories:", df['category'].unique())
    st.write("Follower count range:", df['follower_count'].min(), "-", df['follower_count'].max())

    # Add a 'Show All' option for follower filter
    follower_filter = st.text_input("Follower Filter (e.g. >10000, or leave blank for all)", follower_filter if 'follower_filter' in locals() else "")
    if not follower_filter or follower_filter.strip().lower() == 'show all':
        follower_filter = None

    # Rank and display
    ranked_df = rank_creators(df, category, follower_filter, weight_engagement, weight_followers, weight_likes)
    st.write("Filtered DataFrame:", ranked_df)
    if not ranked_df.empty:
        st.subheader(f"Top {category} creators")
        # Expandable details for each creator
        for idx, row in ranked_df.head(5).iterrows():
            with st.expander(f"{row['name']} (Score: {row['Score']:.2f})"):
                st.write({
                    "Category": row['category'],
                    "Engagement Rate (%)": row['engagement_rate_(%)'],
                    "Follower count": row['follower_count'],
                    "Average likes/comments per post": row['avg_likes_comments'],
                    "Posting frequency": row.get('posting_frequency', 'N/A'),
                    "Historical brand collaborations": row.get('past_brand_collaborations', 'N/A')
                })
        st.markdown("---")
        st.dataframe(
            ranked_df[['name', 'category', 'engagement_rate_(%)', 'follower_count', 'Score']].head(10),
            use_container_width=True
        )
        # Download option
        csv = ranked_df.to_csv(index=False)
        st.download_button("Download Results as CSV", csv, "ranked_creators.csv", "text/csv")
    else:
        st.warning("No creators found after applying filters.")
else:
    st.info("Enter a query to get started.")
