import React, { useState } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Alert,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import { Search } from '@mui/icons-material';
import SentimentChart from '../components/SentimentChart';
import TweetList from '../components/TweetList';
import { searchTweets } from '../services/api';

const SearchPage = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartType, setChartType] = useState('pie');
  const [maxResults, setMaxResults] = useState(100);

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await searchTweets(query, maxResults);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleChartTypeChange = (event, newChartType) => {
    if (newChartType !== null) {
      setChartType(newChartType);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Search & Analyze Tweets
      </Typography>

      {/* Search Controls */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Search Parameters
        </Typography>

        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Search Query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="e.g., bitcoin, #technology, @username"
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
              }}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <TextField
              fullWidth
              type="number"
              label="Max Results"
              value={maxResults}
              onChange={(e) => setMaxResults(Math.min(Math.max(1, parseInt(e.target.value) || 1), 100))}
              inputProps={{ min: 1, max: 100 }}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="contained"
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              startIcon={loading ? <CircularProgress size={20} /> : <Search />}
            >
              Search
            </Button>
          </Grid>
        </Grid>

        {/* Search Tips */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            <strong>Search Tips:</strong> Use keywords, hashtags (#), mentions (@), or phrases.
            You can combine terms with AND, OR, and use quotes for exact phrases.
          </Typography>
        </Box>
      </Paper>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Results */}
      {results && (
        <Grid container spacing={3}>
          {/* Summary Stats */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Search Results Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Typography variant="h4" color="primary">
                    {results.total_count}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Tweets
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="h4" color="success.main">
                    {results.sentiment_stats.counts.positive}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Positive ({results.sentiment_stats.percentages.positive.toFixed(1)}%)
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="h4" color="warning.main">
                    {results.sentiment_stats.counts.neutral}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Neutral ({results.sentiment_stats.percentages.neutral.toFixed(1)}%)
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="h4" color="error.main">
                    {results.sentiment_stats.counts.negative}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Negative ({results.sentiment_stats.percentages.negative.toFixed(1)}%)
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>

          {/* Sentiment Analysis Chart */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Sentiment Analysis
                </Typography>
                <ToggleButtonGroup
                  value={chartType}
                  exclusive
                  onChange={handleChartTypeChange}
                  size="small"
                >
                  <ToggleButton value="pie">
                    Pie
                  </ToggleButton>
                  <ToggleButton value="bar">
                    Bar
                  </ToggleButton>
                </ToggleButtonGroup>
              </Box>
              <SentimentChart
                sentimentStats={results.sentiment_stats}
                chartType={chartType}
              />
            </Paper>
          </Grid>

          {/* Additional Stats */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Analysis Details
              </Typography>
              <Box sx={{ space: 2 }}>
                <Typography variant="body2" gutterBottom>
                  <strong>Average Confidence:</strong> {(results.sentiment_stats.average_confidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>Search Query:</strong> "{query}"
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>Most Positive Sentiment:</strong> {Math.max(
                    results.sentiment_stats.percentages.positive,
                    results.sentiment_stats.percentages.neutral,
                    results.sentiment_stats.percentages.negative
                  ) === results.sentiment_stats.percentages.positive ? 'Positive' :
                    Math.max(
                      results.sentiment_stats.percentages.neutral,
                      results.sentiment_stats.percentages.negative
                    ) === results.sentiment_stats.percentages.neutral ? 'Neutral' : 'Negative'
                  } ({Math.max(
                    results.sentiment_stats.percentages.positive,
                    results.sentiment_stats.percentages.neutral,
                    results.sentiment_stats.percentages.negative
                  ).toFixed(1)}%)
                </Typography>

                {/* Top sentiment examples */}
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Sample Sentiments:
                  </Typography>
                  {['positive', 'neutral', 'negative'].map((sentiment) => {
                    const sampleTweet = results.tweets.find(
                      tweet => tweet.sentiment?.label === sentiment
                    );
                    if (sampleTweet) {
                      return (
                        <Typography key={sentiment} variant="body2" sx={{ mb: 1 }}>
                          <strong>{sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}:</strong> "
                          {sampleTweet.text.substring(0, 100)}
                          {sampleTweet.text.length > 100 ? '...' : ''}"
                        </Typography>
                      );
                    }
                    return null;
                  })}
                </Box>
              </Box>
            </Paper>
          </Grid>

          {/* Tweet List */}
          <Grid item xs={12}>
            <TweetList
              tweets={results.tweets}
              title={`Search Results for "${query}"`}
            />
          </Grid>
        </Grid>
      )}

      {/* No Results State */}
      {!loading && !error && !results && (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Search sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            Search for tweets to analyze sentiment
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Enter keywords, hashtags, or usernames to get started
          </Typography>
        </Box>
      )}
    </Container>
  );
};

export default SearchPage;