import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Australian Social Issues Explorer",
    page_icon="🇦🇺",
    layout="wide"
)

df = pd.read_csv("final_data.csv")

def calculate_severity_score(sentiment_class, score, num_comments):
    """Calculate issue severity: lower sentiment + higher engagement = more severe"""
    severity = (6 - sentiment_class) * (np.log1p(score) + np.log1p(num_comments))
    return severity

def create_quick_stats(df_filtered, category_name, color):
    """Create stats cards for each category"""
    total_posts = len(df_filtered)
    avg_sentiment = df_filtered['senti_class'].mean()
    avg_engagement = df_filtered['score'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"📊 {category_name} Posts",
            f"{total_posts}",
            help=f"Total {category_name.lower()} discussions"
        )
    
    with col2:
        sentiment_color = "🟢" if avg_sentiment > 3 else "🟡" if avg_sentiment > 2.5 else "🔴"
        st.metric(
            f"{sentiment_color} Avg Sentiment",
            f"{avg_sentiment:.1f}/5",
            help="Average sentiment score (1=very negative, 5=very positive)"
        )
    
    with col3:
        st.metric(
            f"👍 Avg Score",
            f"{avg_engagement:.0f}",
            help="Average score (upvotes minus downvotes) per post"
        )

def create_sentiment_overview(df_filtered, color):
    """Create sentiment distribution chart"""
    sentiment_counts = df_filtered['senti_class'].value_counts().sort_index()
    
    fig = px.bar(
        x=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
        y=sentiment_counts.values,
        title="Sentiment Distribution",
        color_discrete_sequence=[color]
    )
    fig.update_layout(
        xaxis_title="Sentiment", 
        yaxis_title="Number of Posts",
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_title_font_color='black',
        yaxis_title_font_color='black'
    )
    return fig

def create_topic_sentiment_chart(df_filtered, color):
    """Main topic sentiment explorer chart"""
    topic_sentiment = df_filtered.groupby(['topic_label', 'senti_class']).size().reset_index(name='count')
    
    # Map sentiment classes to readable labels
    sentiment_map = {1: 'Very Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Very Positive'}
    topic_sentiment['sentiment'] = topic_sentiment['senti_class'].map(sentiment_map)
    
    fig = px.bar(
        topic_sentiment,
        x='topic_label',
        y='count',
        color='sentiment',
        title="Topics by Sentiment Breakdown",
        color_discrete_map={
            'Very Negative': '#d62728',
            'Negative': '#ff7f0e', 
            'Neutral': '#2ca02c',
            'Positive': '#1f77b4',
            'Very Positive': '#9467bd'
        }
    )
    fig.update_xaxes(tickangle=45, title="Topics", 
                     title_font_size=14, title_font_color='black')
    fig.update_yaxes(title="Number of Posts", 
                     title_font_size=14, title_font_color='black')
    fig.update_layout(height=600, legend_title="Sentiment")  # Increased height
    return fig

def create_engagement_sentiment_scatter(df_filtered, color):
    """Engagement vs Sentiment scatter plot"""
    df_filtered['severity'] = calculate_severity_score(
        df_filtered['senti_class'], 
        df_filtered['score'], 
        df_filtered['num_comments']
    )
    
    fig = px.scatter(
        df_filtered,
        x='senti_class',
        y='score',
        size='num_comments',
        hover_data=['topic_label', 'subreddit', 'url'],
        title="Score vs Sentiment",
        color_discrete_sequence=[color]
    )
    fig.update_xaxes(title="Sentiment Score (1=Negative, 5=Positive)", 
                     title_font_size=14, title_font_color='black')
    fig.update_yaxes(title="Score (Upvotes - Downvotes)", 
                     title_font_size=14, title_font_color='black')
    
    # Add explanation for bubble size
    fig.add_annotation(
        text="Each bubble = One Reddit post<br>Bubble size = Number of comments",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color="gray"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    return fig

def create_top_pain_points(df_filtered):
    """Identify most problematic topics"""
    df_filtered['severity'] = calculate_severity_score(
        df_filtered['senti_class'], 
        df_filtered['score'], 
        df_filtered['num_comments']
    )
    
    pain_points = df_filtered.groupby('topic_label').agg({
        'severity': 'mean',
        'senti_class': 'mean',
        'score': 'sum',
        'num_comments': 'sum'
    }).round(2)
    
    pain_points = pain_points.sort_values('severity', ascending=False).head(8)
    pain_points = pain_points.reset_index()
    
    st.markdown("<h3 style='font-size: 20px;'>🚨 Top Pain Points</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; font-style: italic;'>Issues with highest combination of negative sentiment and community engagement</p>", unsafe_allow_html=True)
    
    # Add explanation
    with st.expander("ℹ️ How severity is calculated"):
        st.write("""
        **Severity Score** = (Negative Sentiment) × (Community Engagement)
        
        **Key Definitions:**
        - **Severity Score**: Higher score = More problematic issue that needs attention
        - **Community Engagement**: How much people interact with posts (upvotes + comments)
        - **Score/Upvotes**: Reddit votes showing community interest/agreement
        - **Comments**: Number of people discussing the topic
        - **Sentiment**: 1 (very negative) to 5 (very positive) based on post content
        
        **Why it matters:** Issues with negative sentiment + high engagement indicate 
        community problems that are both serious and widely discussed.
        """)
    
    # Create horizontal bar chart
    fig = px.bar(
        pain_points,
        x='severity',
        y='topic_label',
        orientation='h',
        title="Most Severe Issues (Negative Sentiment + High Engagement)",
        color='senti_class',
        color_continuous_scale='Reds_r',  # Reverse so low sentiment = dark red
        hover_data={
            'score': ':,',  # Format with commas
            'num_comments': ':,',
            'senti_class': ':.1f'
        },
        labels={
            'severity': 'Severity Score',
            'topic_label': 'Discussion Topics',
            'score': 'Total Upvotes',
            'num_comments': 'Total Comments',
            'senti_class': 'Avg Sentiment (1-5)'
        }
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Severity Score (Higher = More Problematic)",
        yaxis_title="Discussion Topics",
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_title_font_color='black',
        yaxis_title_font_color='black',
        coloraxis_colorbar_title="Avg Sentiment<br>(Dark Red=Negative, Light=Positive)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_sample_posts(df_filtered):
    """Show sample Reddit posts for pain point topics"""
    st.markdown("<h3 style='font-size: 20px;'>📝 Sample Reddit Posts</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; font-style: italic;'>See what people are actually saying about these issues</p>", unsafe_allow_html=True)
    
    # Get top 5 pain point topics
    df_filtered['severity'] = calculate_severity_score(
        df_filtered['senti_class'], 
        df_filtered['score'], 
        df_filtered['num_comments']
    )
    
    pain_topics = df_filtered.groupby('topic_label')['severity'].mean().sort_values(ascending=False).head(5)
    
    # Create dropdown for topic selection
    selected_topic = st.selectbox(
        "Select a topic to see sample posts:",
        options=pain_topics.index.tolist(),
        help="Choose a topic to view actual Reddit discussions"
    )
    
    # Get sample posts for selected topic
    topic_posts = df_filtered[df_filtered['topic_label'] == selected_topic].nlargest(3, 'score')
    
    if not topic_posts.empty:
        st.markdown(f"<p style='font-size: 16px;'><strong>Top posts about: {selected_topic}</strong></p>", unsafe_allow_html=True)
        
        for i, (_, post) in enumerate(topic_posts.iterrows(), 1):
            with st.expander(f"Post {i} from r/{post['subreddit']}: Score {post['score']}, {post['num_comments']} comments"):
                # Split title and text
                full_text = post['text_full']
                if '\n' in full_text:
                    title_part = full_text.split('\n')[0]
                    content_part = '\n'.join(full_text.split('\n')[1:])
                else:
                    title_part = full_text
                    content_part = ""
                
                st.markdown(f"<p style='font-size: 15px;'><strong>Title:</strong> {title_part}</p>", unsafe_allow_html=True)
                if content_part.strip():
                    st.markdown(f"<p style='font-size: 15px;'><strong>Content:</strong> {content_part[:500]}{'...' if len(content_part) > 500 else ''}</p>", unsafe_allow_html=True)
                
                # Show additional info
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_label = {1: "😡 Very Negative", 2: "😟 Negative", 3: "😐 Neutral", 
                                     4: "😊 Positive", 5: "😄 Very Positive"}
                    st.markdown(f"<p style='font-size: 15px;'><strong>Sentiment:</strong> {sentiment_label.get(post['senti_class'], 'Unknown')}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 15px;'><strong>Subreddit:</strong> r/{post['subreddit']}</p>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"<p style='font-size: 15px;'><strong>Score:</strong> {post['score']} (upvotes - downvotes)</p>", unsafe_allow_html=True)
                    if 'url' in post and post['url']:
                        st.markdown(f"<p style='font-size: 15px;'><strong><a href='{post['url']}' target='_blank'>View on Reddit</a></strong></p>", unsafe_allow_html=True)
                
    else:
        st.info("No posts found for this topic.")

def create_category_comparison():
    """Compare housing vs cost of living"""
    st.markdown("<h2 style='font-size: 24px;'>🔍 Category Comparison</h2>", unsafe_allow_html=True)
    
    comparison_data = df.groupby('category').agg({
        'senti_class': 'mean',
        'score': 'mean',
        'num_comments': 'mean',
        'text_full': 'count'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=comparison_data.index,
            y=comparison_data['senti_class'],
            title="Average Sentiment by Category",
            color=comparison_data.index,
            color_discrete_map={'housing': '#2E86AB', 'cost_of_living': '#F24236'}
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Average Sentiment Score",
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            xaxis_title_font_color='black',
            yaxis_title_font_color='black'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=comparison_data.index,
            y=comparison_data['score'],
            title="Average Score by Category",
            color=comparison_data.index,
            color_discrete_map={'housing': '#2E86AB', 'cost_of_living': '#F24236'}
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Average Score (Upvotes - Downvotes)",
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            xaxis_title_font_color='black',
            yaxis_title_font_color='black'
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🇦🇺 Australian Social Issues Explorer")
    st.markdown("<h3 style='text-align: center; color: #666; font-size: 18px;'>Analysing Housing and Cost of Living Discussions</h3>", unsafe_allow_html=True)
    
    # Add key definitions at the top
    with st.expander("📖 Key Definitions"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div style='font-size: 16px;'>
            <strong>📊 Metrics Explained:</strong><br>
            • <strong>Score:</strong> Reddit upvotes minus downvotes (can be negative if heavily downvoted)<br>
            • <strong>Upvote Ratio:</strong> Percentage of votes that are upvotes (e.g., 0.85 = 85% upvoted)<br>
            • <strong>Comments:</strong> Number of replies - indicates discussion volume<br>
            • <strong>Community Engagement:</strong> Total interaction (score + comments)<br>
            • <strong>Sentiment Score:</strong> 1 (very negative) to 5 (very positive) based on post content<br>
            • <strong>Topics:</strong> Discussion themes identified from post content (e.g., "rental prices", "grocery costs")<br>
            • <strong>Severity:</strong> Combines negative sentiment with high engagement to identify urgent issues<br>
            • <strong>Subreddits:</strong> Which Australian Reddit communities are discussing these topics
            </div>
            """, unsafe_allow_html=True)
        with col2:
            # Add date information prominently
            if 'created_datetime' in df.columns:
                df['created_datetime'] = pd.to_datetime(df['created_datetime'])
                earliest = df['created_datetime'].min().strftime('%d %b %Y')
                latest = df['created_datetime'].max().strftime('%d %b %Y')
                days_span = (df['created_datetime'].max() - df['created_datetime'].min()).days
                st.markdown(f"""
                <div style='font-size: 16px;'>
                <strong>📅 Data Period:</strong><br><br>
                <strong>From:</strong> {earliest}<br><br>
                <strong>To:</strong> {latest}<br><br>
                <strong>Duration:</strong> {days_span} days<br><br>
                <strong>Collection Date:</strong> 03-07-2025<br><br>
                <strong>Total Posts:</strong> {len(df)}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<p style='font-size: 16px;'><strong>🎯 What This Shows:</strong> Which housing and cost-of-living issues Australians discuss most and feel most strongly about during this period.</p>", unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Category filter
    view_option = st.sidebar.selectbox(
        "Select View:",
        ["Housing Analysis", "Cost of Living Analysis", "Compare Both"]
    )
    
    # Methodology section
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Methodology")
    with st.sidebar.expander("How this dashboard was created"):
        st.markdown("""
        **1. Data Collection**
        - Posts collected from selected Australian subreddits
        - Collection Date: 03-07-2025
        - Focus on housing and cost of living discussions
        
        **2. Category Classification**
        - Zero-shot classification using OpenAI GPT-4o-mini
        - Posts classified as 'housing', 'cost_of_living', or 'other'
        - Filtered out posts not belonging to housing/cost categories
        
        **3. Topic Modeling**
        - BERTopic algorithm for topic discovery
        - OpenAI GPT-4o-mini for generating topic labels
        - Each post assigned to most relevant topic
        
        **4. Sentiment Analysis**
        - Hugging Face transformer model
        - Sentiment scored 1 (very negative) to 5 (very positive)
        
        **5. Analysis Metrics**
        - Severity Score: Combines negative sentiment + engagement
        - Engagement: Reddit score + comment volume
        """, unsafe_allow_html=True)
    
    # Apply filters - removed sentiment filter
    df_filtered = df.copy()
    
    if view_option == "Compare Both":
        create_category_comparison()
        
        # Show both categories side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #2E86AB; font-size: 20px;'>🏠 Housing Issues</h3>", unsafe_allow_html=True)
            housing_df = df_filtered[df_filtered['category'] == 'housing']
            if not housing_df.empty:
                create_quick_stats(housing_df, "Housing", "#2E86AB")
                st.plotly_chart(create_sentiment_overview(housing_df, "#2E86AB"), use_container_width=True)
        
        with col2:
            st.markdown("<h3 style='color: #F24236; font-size: 20px;'>💰 Cost of Living Issues</h3>", unsafe_allow_html=True)
            cost_df = df_filtered[df_filtered['category'] == 'cost_of_living']
            if not cost_df.empty:
                create_quick_stats(cost_df, "Cost of Living", "#F24236")
                st.plotly_chart(create_sentiment_overview(cost_df, "#F24236"), use_container_width=True)
    
    elif view_option == "Housing Analysis":
        st.markdown("<h2 style='color: #2E86AB; font-size: 24px;'>🏠 Housing Issues Analysis</h2>", unsafe_allow_html=True)
        housing_df = df_filtered[df_filtered['category'] == 'housing']
        
        if housing_df.empty:
            st.warning("No housing data matches your filters.")
            return
        
        # Quick stats
        create_quick_stats(housing_df, "Housing", "#2E86AB")
        
        st.markdown("---")
        
        # Main visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_sentiment_overview(housing_df, "#2E86AB"), use_container_width=True)
            
        with col2:
            st.plotly_chart(create_engagement_sentiment_scatter(housing_df, "#2E86AB"), use_container_width=True)
        
        # Topic sentiment breakdown
        st.plotly_chart(create_topic_sentiment_chart(housing_df, "#2E86AB"), use_container_width=True)
        
        # Pain points
        create_top_pain_points(housing_df)
        
        # Sample posts
        show_sample_posts(housing_df)
        
        # Subreddit breakdown
        st.markdown("<h3 style='font-size: 20px;'>🗂️ Discussion Sources</h3>", unsafe_allow_html=True)
        subreddit_counts = housing_df['subreddit'].value_counts().head(6)
        fig_sub = px.pie(
            values=subreddit_counts.values,
            names=[f"r/{sub}" for sub in subreddit_counts.index],
            title="Housing Discussions by Subreddit"
        )
        fig_sub.update_layout(
            title_font_size=16,
            title_font_color='black'
        )
        st.plotly_chart(fig_sub, use_container_width=True)
    
    elif view_option == "Cost of Living Analysis":
        st.markdown("<h2 style='color: #F24236; font-size: 24px;'>💰 Cost of Living Issues Analysis</h2>", unsafe_allow_html=True)
        cost_df = df_filtered[df_filtered['category'] == 'cost_of_living']
        
        if cost_df.empty:
            st.warning("No cost of living data matches your filters.")
            return
        
        # Quick stats
        create_quick_stats(cost_df, "Cost of Living", "#F24236")
        
        st.markdown("---")
        
        # Main visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_sentiment_overview(cost_df, "#F24236"), use_container_width=True)
            
        with col2:
            st.plotly_chart(create_engagement_sentiment_scatter(cost_df, "#F24236"), use_container_width=True)
        
        # Topic sentiment breakdown
        st.plotly_chart(create_topic_sentiment_chart(cost_df, "#F24236"), use_container_width=True)
        
        # Pain points
        create_top_pain_points(cost_df)
        
        # Sample posts
        show_sample_posts(cost_df)
        
        # Subreddit breakdown
        st.markdown("<h3 style='font-size: 20px;'>🗂️ Discussion Sources</h3>", unsafe_allow_html=True)
        subreddit_counts = cost_df['subreddit'].value_counts().head(6)
        fig_sub = px.pie(
            values=subreddit_counts.values,
            names=[f"r/{sub}" for sub in subreddit_counts.index],
            title="Cost of Living Discussions by Subreddit"
        )
        fig_sub.update_layout(
            title_font_size=16,
            title_font_color='black'
        )
        st.plotly_chart(fig_sub, use_container_width=True)
    
    # Data summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Data Summary**")
    st.sidebar.write(f"Total Posts: {len(df_filtered)}")
    st.sidebar.write(f"Housing: {len(df_filtered[df_filtered['category'] == 'housing'])}")
    st.sidebar.write(f"Cost of Living: {len(df_filtered[df_filtered['category'] == 'cost_of_living'])}")
    st.sidebar.write(f"Unique Topics: {df_filtered['topic_label'].nunique()}")
    
    # Add date range information
    if 'created_datetime' in df.columns:
        df['created_datetime'] = pd.to_datetime(df['created_datetime'])
        earliest_date = df['created_datetime'].min().strftime('%Y-%m-%d')
        latest_date = df['created_datetime'].max().strftime('%Y-%m-%d')
        st.sidebar.markdown("**📅 Data Period**")
        st.sidebar.write(f"From: {earliest_date}")
        st.sidebar.write(f"To: {latest_date}")
        st.sidebar.write(f"Span: {(df['created_datetime'].max() - df['created_datetime'].min()).days} days")
        st.sidebar.write("**Collection:** 03-07-2025")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Australian Social Issues Explorer</strong></p>
        <p>Built with Streamlit • Data from Reddit • Topic Modeling & Sentiment Analysis</p>
        <p><em>Understanding Australian community concerns through data</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()