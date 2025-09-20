import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

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
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
                     title_font_size=14)
    fig.update_yaxes(title="Number of Posts", 
                     title_font_size=14)
    fig.update_layout(
        height=600, 
        legend_title="Sentiment",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_engagement_sentiment_scatter(df_filtered, color):
    """Engagement vs Sentiment scatter plot with improved visibility and explanation"""
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
        title="Community Response: How Australians Vote on Different Sentiment Posts",
        color_discrete_sequence=[color]
    )
    fig.update_xaxes(title="Sentiment Score (1=Negative, 5=Positive)", 
                     title_font_size=14)
    fig.update_yaxes(title="Score (Upvotes - Downvotes)", 
                     title_font_size=14)
    
    # Add quadrant explanation
    fig.add_annotation(
        text="📍 Bottom-left = Widespread community problems<br>📍 Top-right = Community solutions/success stories<br>📍 Bubble size = Number of comments",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color="#333333"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#333333",
        borderwidth=1
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_trending_analysis(df_filtered):
    """Show trending topics over time"""
    st.markdown("<h3 style='font-size: 20px;'>📈 Discussion Volume Trends</h3>", unsafe_allow_html=True)
    
    if 'created_datetime' not in df_filtered.columns:
        st.info("Time-based analysis requires datetime data. Please ensure 'created_datetime' column exists.")
        return
    
    # Convert to datetime and create weekly aggregation
    df_filtered['created_datetime'] = pd.to_datetime(df_filtered['created_datetime'])
    df_filtered['week'] = df_filtered['created_datetime'].dt.to_period('W').dt.start_time
    
    # Get top 5 topics by volume
    top_topics = df_filtered['topic_label'].value_counts().head(5).index
    
    # Create weekly trend data for discussion volume only
    weekly_trends = df_filtered[df_filtered['topic_label'].isin(top_topics)].groupby(['week', 'topic_label']).size().reset_index(name='post_count')
    
    # Create simple line chart for volume trends
    fig = px.line(
        weekly_trends,
        x='week',
        y='post_count',
        color='topic_label',
        title="Weekly Discussion Volume for Top Issues",
        labels={'post_count': 'Number of Posts', 'week': 'Week', 'topic_label': 'Topic'}
    )
    
    fig.update_layout(
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Issue intensity analysis
    st.markdown("<h4 style='font-size: 18px;'>🚨 Issue Intensity Analysis</h4>", unsafe_allow_html=True)
    
    # Calculate recent vs historical patterns (using last 25% of data as "recent")
    df_sorted = df_filtered.sort_values('created_datetime')
    recent_cutoff = len(df_sorted) * 0.75
    recent_data = df_sorted.iloc[int(recent_cutoff):]
    historical_data = df_sorted.iloc[:int(recent_cutoff)]
    
    if len(recent_data) > 0 and len(historical_data) > 0:
        recent_summary = recent_data.groupby('topic_label').agg({
            'senti_class': 'mean',
            'text_full': 'count'
        })
        
        historical_summary = historical_data.groupby('topic_label').agg({
            'senti_class': 'mean',
            'text_full': 'count'
        })
        
        # Find topics with significant changes
        intensity_alerts = []
        for topic in recent_summary.index:
            if topic in historical_summary.index and recent_summary.loc[topic, 'text_full'] >= 3:
                sentiment_change = recent_summary.loc[topic, 'senti_class'] - historical_summary.loc[topic, 'senti_class']
                volume_ratio = recent_summary.loc[topic, 'text_full'] / max(historical_summary.loc[topic, 'text_full'], 1)
                
                # Flag if sentiment dropped significantly OR volume increased significantly
                if sentiment_change < -0.3 or volume_ratio > 1.5:
                    intensity_alerts.append({
                        'topic': topic,
                        'sentiment_change': sentiment_change,
                        'volume_ratio': volume_ratio,
                        'recent_posts': recent_summary.loc[topic, 'text_full']
                    })
        
        if intensity_alerts:
            intensity_alerts = sorted(intensity_alerts, key=lambda x: abs(x['sentiment_change']) + x['volume_ratio'], reverse=True)
            
            for alert in intensity_alerts[:3]:  # Show top 3 alerts
                if alert['sentiment_change'] < -0.3:
                    st.warning(f"📉 **{alert['topic']}**: Sentiment decreased by {abs(alert['sentiment_change']):.1f} points in recent period ({alert['recent_posts']} recent posts)")
                elif alert['volume_ratio'] > 1.5:
                    st.info(f"📈 **{alert['topic']}**: Discussion volume increased {alert['volume_ratio']:.1f}x in recent period ({alert['recent_posts']} recent posts)")
        else:
            st.success("✅ No significant issue intensity changes detected in recent period")
    else:
        st.info("Insufficient data to compare recent vs historical patterns")

def create_geographic_analysis(df_filtered):
    """Analyze issues by location if location data available"""
    st.markdown("<h3 style='font-size: 20px;'>🗺️ Geographic Issue Distribution</h3>", unsafe_allow_html=True)
    
    # Check if we have location indicators in subreddit names
    location_keywords = {
        'sydney': ['sydney', 'nsw'],
        'melbourne': ['melbourne', 'victoria'],
        'brisbane': ['brisbane', 'queensland'],
        'perth': ['perth', 'westernaustralia'],
        'adelaide': ['adelaide', 'southaustralia'],
        'australia_general': ['australia', 'aus', 'aussie']
    }
    
    # Map subreddits to locations
    def get_location(subreddit):
        subreddit_lower = subreddit.lower()
        for location, keywords in location_keywords.items():
            if any(keyword in subreddit_lower for keyword in keywords):
                return location
        return 'other'
    
    df_filtered['location'] = df_filtered['subreddit'].apply(get_location)
    
    # Simple location summary instead of complex faceted chart
    location_summary = df_filtered.groupby('location').agg({
        'senti_class': 'mean',
        'score': 'mean',
        'text_full': 'count'
    }).round(2)
    
    # Filter out 'other' if it exists
    location_summary = location_summary[location_summary.index != 'other']
    
    if not location_summary.empty and len(location_summary) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                x=location_summary.index,
                y=location_summary['text_full'],
                title="Discussion Volume by Region",
                color=location_summary.index,
                labels={'y': 'Number of Posts', 'x': 'Region'}
            )
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                x=location_summary.index,
                y=location_summary['senti_class'],
                title="Average Sentiment by Region",
                color=location_summary['senti_class'],
                color_continuous_scale='RdYlGn',
                labels={'y': 'Average Sentiment', 'x': 'Region'}
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Regional insights
        st.markdown("<h4 style='font-size: 18px;'>📍 Regional Insights</h4>", unsafe_allow_html=True)
        
        if len(location_summary) > 1:
            most_positive = location_summary['senti_class'].idxmax()
            least_positive = location_summary['senti_class'].idxmin()
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🟢 **Most Optimistic**: {most_positive.title().replace('_', ' ')} (Sentiment: {location_summary.loc[most_positive, 'senti_class']:.1f})")
            with col2:
                st.error(f"🔴 **Most Concerned**: {least_positive.title().replace('_', ' ')} (Sentiment: {location_summary.loc[least_positive, 'senti_class']:.1f})")
    else:
        st.info("Geographic analysis requires location-specific subreddits. Current data appears to be from general Australian communities.")

def create_solution_identification(df_filtered):
    """Find highly upvoted positive posts - potential solutions organized by topic"""
    st.markdown("<h3 style='font-size: 20px;'>💡 Community Solutions & Success Stories</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; font-style: italic;'>Topics with highly upvoted positive posts that might contain solutions</p>", unsafe_allow_html=True)
    
    # Find positive posts (sentiment >= 4) with high engagement
    solutions = df_filtered[
        (df_filtered['senti_class'] >= 4) & 
        (df_filtered['score'] >= df_filtered['score'].quantile(0.7))
    ]
    
    if not solutions.empty:
        # Group by topic and get stats
        solution_topics = solutions.groupby('topic_label').agg({
            'score': ['mean', 'count'],
            'senti_class': 'mean'
        }).round(2)
        
        # Flatten column names
        solution_topics.columns = ['avg_score', 'post_count', 'avg_sentiment']
        solution_topics = solution_topics[solution_topics['post_count'] >= 2]  # At least 2 positive posts
        solution_topics = solution_topics.sort_values('avg_score', ascending=False)
        
        if not solution_topics.empty:
            # Show topic overview chart
            fig = px.bar(
                x=solution_topics.index,
                y=solution_topics['avg_score'],
                title="Topics with Most Positive Community Response",
                color=solution_topics['avg_sentiment'],
                color_continuous_scale='Greens',
                labels={'y': 'Average Score', 'x': 'Topics with Positive Solutions'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic selection and posts
            st.markdown("<h4 style='font-size: 18px;'>✨ Explore Solution Posts by Topic</h4>", unsafe_allow_html=True)
            
            selected_solution_topic = st.selectbox(
                "Select a topic to see positive posts:",
                options=solution_topics.index.tolist(),
                help="Choose a topic to view community solutions and success stories"
            )
            
            # Get posts for selected topic
            topic_solution_posts = solutions[
                solutions['topic_label'] == selected_solution_topic
            ].nlargest(5, 'score')
            
            if not topic_solution_posts.empty:
                # Show topic stats
                topic_stats = topic_solution_posts.agg({
                    'score': 'mean',
                    'senti_class': 'mean',
                    'num_comments': 'mean'
                })
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Solutions Found", len(topic_solution_posts))
                with col2:
                    st.metric("Avg Score", f"{topic_stats['score']:.0f}")
                with col3:
                    st.metric("Avg Sentiment", f"{topic_stats['senti_class']:.1f}/5")
                
                st.markdown(f"<p style='font-size: 16px;'><strong>Positive posts about: {selected_solution_topic}</strong></p>", unsafe_allow_html=True)
                
                for i, (_, post) in enumerate(topic_solution_posts.iterrows(), 1):
                    with st.expander(f"💡 Solution {i}: Score {post['score']}, r/{post['subreddit']} ({post['num_comments']} comments)"):
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
                            st.markdown(f"<p style='font-size: 15px;'><strong>Content:</strong> {content_part[:400]}{'...' if len(content_part) > 400 else ''}</p>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"<p style='font-size: 15px;'><strong>Sentiment:</strong> 😊 Very Positive ({post['senti_class']:.1f}/5)</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='font-size: 15px;'><strong>Engagement:</strong> {post['num_comments']} comments</p>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"<p style='font-size: 15px;'><strong>Score:</strong> {post['score']} upvotes</p>", unsafe_allow_html=True)
                            if 'url' in post and post['url']:
                                st.markdown(f"<p style='font-size: 15px;'><strong><a href='{post['url']}' target='_blank'>Read Full Post</a></strong></p>", unsafe_allow_html=True)
            else:
                st.info("No solution posts found for this topic.")
        else:
            st.info("No topics found with multiple positive, highly-rated posts.")
    else:
        st.info("No highly positive posts found in current filter selection.")

def create_top_pain_points(df_filtered):
    """Identify most problematic topics with improved subtitle"""
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
    st.markdown("<p style='font-size: 14px; color: #888;'><strong>Ranked by severity (highest to lowest)</strong></p>", unsafe_allow_html=True)
    
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
        coloraxis_colorbar_title="Avg Sentiment<br>(Dark Red=Negative, Light=Positive)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_sample_posts(df_filtered):
    """Show sample Reddit posts for all topics with improved dropdown styling"""
    st.markdown("<h3 style='font-size: 20px;'>📝 Sample Reddit Posts</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; font-style: italic;'>See what people are actually saying about these issues</p>", unsafe_allow_html=True)
    
    # Get ALL unique topics, sorted by frequency
    all_topics = df_filtered['topic_label'].value_counts().index.tolist()
    
    # Add custom CSS for dropdown styling
    st.markdown("""
    <style>
    .stSelectbox > div > div > select {
        background: linear-gradient(45deg, #f0f2f6, #e6e9ef) !important;
        color: #262730 !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div > select option {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
        padding: 8px !important;
        margin: 2px 0 !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div > select option:nth-child(even) {
        background: linear-gradient(45deg, #2196F3, #1976D2) !important;
    }
    .stSelectbox > div > div > select option:nth-child(3n) {
        background: linear-gradient(45deg, #FF9800, #F57C00) !important;
    }
    .stSelectbox > div > div > select option:nth-child(4n) {
        background: linear-gradient(45deg, #9C27B0, #7B1FA2) !important;
    }
    .stSelectbox > div > div > select option:nth-child(5n) {
        background: linear-gradient(45deg, #E91E63, #C2185B) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create dropdown for topic selection with ALL topics
    selected_topic = st.selectbox(
        "Select a topic to see sample posts:",
        options=all_topics,
        help="Choose any topic to view actual Reddit discussions (all topics available)"
    )
    
    # Calculate severity for context
    df_filtered['severity'] = calculate_severity_score(
        df_filtered['senti_class'], 
        df_filtered['score'], 
        df_filtered['num_comments']
    )
    
    # Get sample posts for selected topic
    topic_posts = df_filtered[df_filtered['topic_label'] == selected_topic].nlargest(3, 'score')
    
    if not topic_posts.empty:
        # Show topic statistics
        topic_stats = df_filtered[df_filtered['topic_label'] == selected_topic]
        avg_sentiment = topic_stats['senti_class'].mean()
        total_posts = len(topic_stats)
        avg_score = topic_stats['score'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Posts", total_posts)
        with col2:
            sentiment_emoji = "😡" if avg_sentiment < 2 else "😟" if avg_sentiment < 3 else "😐" if avg_sentiment < 4 else "😊"
            st.metric("Avg Sentiment", f"{avg_sentiment:.1f}/5", help=f"{sentiment_emoji}")
        with col3:
            st.metric("Avg Score", f"{avg_score:.0f}")
        
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

def create_community_insights(df_filtered):
    """Analyze which subreddits discuss which issues"""
    st.markdown("<h3 style='font-size: 20px;'>🏘️ Community Discussion Patterns</h3>", unsafe_allow_html=True)
    
    # Create subreddit-topic matrix
    community_topics = df_filtered.groupby(['subreddit', 'topic_label']).agg({
        'text_full': 'count',
        'senti_class': 'mean'
    }).reset_index()
    
    # Filter for meaningful volumes
    community_topics = community_topics[community_topics['text_full'] >= 2]
    
    # Get top subreddits and topics
    top_subreddits = df_filtered['subreddit'].value_counts().head(8).index
    top_topics = df_filtered['topic_label'].value_counts().head(10).index
    
    filtered_data = community_topics[
        (community_topics['subreddit'].isin(top_subreddits)) &
        (community_topics['topic_label'].isin(top_topics))
    ]
    
    if not filtered_data.empty:
        # Create heatmap showing which communities discuss which topics
        pivot_data = filtered_data.pivot(index='subreddit', columns='topic_label', values='text_full').fillna(0)
        
        fig = px.imshow(
            pivot_data.values,
            x=pivot_data.columns,
            y=[f"r/{sub}" for sub in pivot_data.index],
            title="Community-Topic Discussion Matrix",
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig.update_layout(
            xaxis_title="Discussion Topics",
            yaxis_title="Subreddit Communities",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Community specializations
        st.markdown("<h4 style='font-size: 18px;'>🎯 Community Specializations</h4>", unsafe_allow_html=True)
        
        # Find which communities are most focused on specific topics
        specializations = []
        for subreddit in top_subreddits:
            subreddit_data = df_filtered[df_filtered['subreddit'] == subreddit]
            if len(subreddit_data) > 5:  # Minimum posts for meaningful analysis
                top_topic = subreddit_data['topic_label'].value_counts().iloc[0]
                topic_name = subreddit_data['topic_label'].value_counts().index[0]
                total_posts = len(subreddit_data)
                specialization_pct = (top_topic / total_posts) * 100
                
                if specialization_pct > 30:  # At least 30% focused on one topic
                    specializations.append({
                        'subreddit': subreddit,
                        'topic': topic_name,
                        'percentage': specialization_pct,
                        'posts': top_topic
                    })
        
        if specializations:
            for spec in sorted(specializations, key=lambda x: x['percentage'], reverse=True)[:5]:
                st.info(f"🎯 **r/{spec['subreddit']}** specializes in **{spec['topic']}** ({spec['percentage']:.0f}% of posts, {spec['posts']} discussions)")
        else:
            st.info("Most communities discuss a diverse range of topics rather than specializing.")

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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
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
                <strong>Collection Date:</strong> 16-09-2025<br><br>
                <strong>Total Posts:</strong> {len(df)}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<p style='font-size: 16px;'><strong>🎯 What This Shows:</strong> Which housing and cost-of-living issues Australians discuss most and feel most strongly about during this period.</p>", unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Category filter
    view_option = st.sidebar.selectbox(
        "Select View:",
        ["Housing Analysis", "Cost of Living Analysis", "Compare Both", "Trending Analysis", "Geographic Analysis", "Solution Finder"]
    )
    
    # Methodology section
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Methodology")
    with st.sidebar.expander("How this dashboard was created"):
        st.markdown("""
        **1. Data Collection**
        - Posts collected from selected Australian subreddits
        - Collection Date: 16-09-2025
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
    
    # Apply filters
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
    
    elif view_option == "Trending Analysis":
        st.markdown("<h2 style='color: #9C27B0; font-size: 24px;'>📈 Trending Issues Analysis</h2>", unsafe_allow_html=True)
        create_trending_analysis(df_filtered)
        
        # Add community insights
        create_community_insights(df_filtered)
    
    elif view_option == "Geographic Analysis":
        st.markdown("<h2 style='color: #4CAF50; font-size: 24px;'>🗺️ Geographic Analysis</h2>", unsafe_allow_html=True)
        create_geographic_analysis(df_filtered)
    
    elif view_option == "Solution Finder":
        st.markdown("<h2 style='color: #FF9800; font-size: 24px;'>💡 Solution Finder</h2>", unsafe_allow_html=True)
        create_solution_identification(df_filtered)
    
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
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
        st.sidebar.write("**Collection:** 16-09-2025")
    
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
