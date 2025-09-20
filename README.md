# 🇦🇺 Australian Social Issues Explorer

An interactive dashboard analyzing Australian housing and cost of living discussions from Reddit communities, providing insights into community sentiment and emerging social issues.

## 📊 Overview

This Streamlit application analyzes Reddit posts from Australian communities to understand public sentiment around housing and cost of living issues. The dashboard identifies trending topics, community concerns, and potential solutions through data-driven analysis.

## ✨ Features

### 🏠 Housing Analysis
- Sentiment distribution across housing discussions
- Community engagement patterns
- Pain point identification with severity scoring
- Sample Reddit posts for context

### 💰 Cost of Living Analysis
- Cost of living sentiment trends
- Topic breakdown by sentiment
- Regional discussion patterns
- Community response analysis

### 📈 Trending Analysis
- Weekly discussion volume trends
- Issue intensity analysis
- Historical vs recent period comparisons
- Community discussion patterns

### 🗺️ Geographic Analysis
- Regional sentiment distribution
- Discussion volume by location
- Most/least optimistic regions
- Location-based insights

### 💡 Solution Finder
- Highly upvoted positive posts
- Community solutions by topic
- Success stories and helpful discussions
- Topic-based solution exploration

### 📝 Sample Posts Explorer
- Browse actual Reddit discussions
- All topics available with enhanced dropdown
- Post statistics and sentiment context
- Direct links to original Reddit posts

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Visualization**: Plotly Express
- **Data Processing**: Pandas, NumPy
- **Sentiment Analysis**: Hugging Face Transformers
- **Topic Modeling**: BERTopic
- **Classification**: OpenAI GPT-4o-mini

## 📅 Data Information

- **Collection Period**: June 9, 2025 - September 16, 2025 (99 days)
- **Collection Date**: September 16, 2025
- **Data Source**: Australian Reddit communities
- **Categories**: Housing and Cost of Living discussions
- **Processing**: Automated sentiment analysis and topic classification

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas plotly numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pkarcreative/reddit_social_issues.git
cd reddit_social_issues
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
reddit_social_issues/
├── app.py                 # Main Streamlit application
├── final_data.csv         # Processed Reddit data
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🔧 Data Processing Pipeline

1. **Data Collection**: Posts collected from selected Australian subreddits
2. **Category Classification**: Zero-shot classification using GPT-4o-mini
3. **Topic Modeling**: BERTopic algorithm for topic discovery
4. **Sentiment Analysis**: Transformer-based sentiment scoring (1-5 scale)
5. **Severity Calculation**: Combines negative sentiment with engagement metrics

## 📊 Key Metrics Explained

- **Score**: Reddit upvotes minus downvotes
- **Sentiment Score**: 1 (very negative) to 5 (very positive)
- **Severity Score**: Combines negative sentiment with high engagement
- **Community Engagement**: Total interaction (score + comments)
- **Topics**: Discussion themes identified through topic modeling

## 🎯 Use Cases

### For Researchers
- Understanding Australian social sentiment trends
- Identifying emerging community concerns
- Academic research on social media discourse

### For Policymakers
- Public opinion monitoring
- Issue priority identification
- Community feedback analysis

### For Community Leaders
- Understanding local concerns
- Finding community solutions
- Tracking discussion patterns

### For General Public
- Awareness of current social issues
- Understanding community sentiment
- Finding relevant discussions and solutions

## ⚠️ Limitations

- **Time Period**: Data covers only 99 days (June-September 2025)
- **Platform Specific**: Limited to Reddit discussions
- **Decision Making**: Not suitable for individual financial decisions
- **Geographic Scope**: Based on subreddit location indicators
- **Sample Bias**: Represents Reddit users, not general population

## 🔮 Future Enhancements

- Extended data collection (6+ months for trend analysis)
- Integration with economic indicators
- Demographic analysis capabilities
- Real-time data updates
- Enhanced geographic granularity
- Predictive modeling features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Reddit communities for the discussion data
- Streamlit for the web framework
- Plotly for interactive visualizations
- Hugging Face for sentiment analysis models
- OpenAI for classification capabilities

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com]

---

**Note**: This dashboard provides insights into community sentiment and discussions. It is not intended for financial or investment decision-making. Always consult professional advisors for financial decisions.
