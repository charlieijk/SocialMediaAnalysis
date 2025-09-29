import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  TextField,
  Chip,
  Box,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Alert,
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import SentimentChart from './SentimentChart';
import TweetList from './TweetList';
import { startStreaming, stopStreaming, getStreamingData } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const RealTimeDashboard = () => {
  const [streamingData, setStreamingData] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [keywords, setKeywords] = useState([]);
  const [newKeyword, setNewKeyword] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    let interval;
    if (autoRefresh && isStreaming) {
      interval = setInterval(fetchStreamingData, 5000); // Refresh every 5 seconds
    }
    return () => clearInterval(interval);
  }, [autoRefresh, isStreaming]);

  const fetchStreamingData = async () => {
    try {
      const data = await getStreamingData();
      setStreamingData(data);
      setIsStreaming(data.is_streaming);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStartStreaming = async () => {
    if (keywords.length === 0) {
      setError('Please add at least one keyword');
      return;
    }

    try {
      await startStreaming(keywords);
      setIsStreaming(true);
      setSuccess('Streaming started successfully');
      setError(null);
      fetchStreamingData();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStopStreaming = async () => {
    try {
      await stopStreaming();
      setIsStreaming(false);
      setSuccess('Streaming stopped successfully');
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleAddKeyword = () => {
    if (newKeyword.trim() && !keywords.includes(newKeyword.trim())) {
      setKeywords([...keywords, newKeyword.trim()]);
      setNewKeyword('');
    }
  };

  const handleRemoveKeyword = (keywordToRemove) => {
    setKeywords(keywords.filter(k => k !== keywordToRemove));
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAddKeyword();
    }
  };

  // Prepare sentiment timeline chart data
  const getSentimentTimelineData = () => {
    if (!streamingData?.sentiment_history || streamingData.sentiment_history.length === 0) {
      return null;
    }

    const labels = streamingData.sentiment_history.map(point => {
      const date = new Date(point.timestamp);
      return date.toLocaleTimeString();
    });

    return {
      labels,
      datasets: [
        {
          label: 'Positive',
          data: streamingData.sentiment_history.map(point => point.sentiment_counts.positive),
          borderColor: '#4CAF50',
          backgroundColor: 'rgba(76, 175, 80, 0.1)',
          tension: 0.1,
        },
        {
          label: 'Neutral',
          data: streamingData.sentiment_history.map(point => point.sentiment_counts.neutral),
          borderColor: '#FFC107',
          backgroundColor: 'rgba(255, 193, 7, 0.1)',
          tension: 0.1,
        },
        {
          label: 'Negative',
          data: streamingData.sentiment_history.map(point => point.sentiment_counts.negative),
          borderColor: '#F44336',
          backgroundColor: 'rgba(244, 67, 54, 0.1)',
          tension: 0.1,
        },
      ],
    };
  };

  const timelineOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Real-time Sentiment Trends',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  // Calculate current sentiment stats
  const getCurrentSentimentStats = () => {
    if (!streamingData?.tweets || streamingData.tweets.length === 0) {
      return null;
    }

    const sentimentCounts = { positive: 0, neutral: 0, negative: 0 };
    streamingData.tweets.forEach(tweet => {
      const sentiment = tweet.sentiment?.label || 'neutral';
      sentimentCounts[sentiment]++;
    });

    const total = streamingData.tweets.length;
    const percentages = {
      positive: (sentimentCounts.positive / total) * 100,
      neutral: (sentimentCounts.neutral / total) * 100,
      negative: (sentimentCounts.negative / total) * 100,
    };

    return { counts: sentimentCounts, percentages, total_tweets: total };
  };

  const sentimentTimelineData = getSentimentTimelineData();
  const currentSentimentStats = getCurrentSentimentStats();

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Real-time Social Media Sentiment Analysis
      </Typography>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {/* Controls */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Streaming Controls
        </Typography>

        <Box sx={{ mb: 2 }}>
          <TextField
            label="Add keyword"
            value={newKeyword}
            onChange={(e) => setNewKeyword(e.target.value)}
            onKeyPress={handleKeyPress}
            sx={{ mr: 1 }}
          />
          <Button variant="outlined" onClick={handleAddKeyword}>
            Add
          </Button>
        </Box>

        <Box sx={{ mb: 2 }}>
          {keywords.map((keyword) => (
            <Chip
              key={keyword}
              label={keyword}
              onDelete={() => handleRemoveKeyword(keyword)}
              sx={{ mr: 1, mb: 1 }}
            />
          ))}
        </Box>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleStartStreaming}
            disabled={isStreaming || keywords.length === 0}
          >
            Start Streaming
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleStopStreaming}
            disabled={!isStreaming}
          >
            Stop Streaming
          </Button>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto-refresh"
          />
          <Typography variant="body2" color={isStreaming ? 'success.main' : 'text.secondary'}>
            Status: {isStreaming ? 'Streaming' : 'Stopped'}
          </Typography>
        </Box>
      </Paper>

      {/* Dashboard Grid */}
      <Grid container spacing={3}>
        {/* Current Stats */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Session Stats
              </Typography>
              {streamingData && (
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Total Tweets: {streamingData.tweets?.length || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Last Update: {streamingData.last_update ?
                      new Date(streamingData.last_update).toLocaleTimeString() : 'Never'}
                  </Typography>
                  {currentSentimentStats && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        Positive: {currentSentimentStats.counts.positive} ({currentSentimentStats.percentages.positive.toFixed(1)}%)
                      </Typography>
                      <Typography variant="body2">
                        Neutral: {currentSentimentStats.counts.neutral} ({currentSentimentStats.percentages.neutral.toFixed(1)}%)
                      </Typography>
                      <Typography variant="body2">
                        Negative: {currentSentimentStats.counts.negative} ({currentSentimentStats.percentages.negative.toFixed(1)}%)
                      </Typography>
                    </Box>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Sentiment Distribution */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            {currentSentimentStats ? (
              <SentimentChart sentimentStats={currentSentimentStats} chartType="pie" />
            ) : (
              <Box sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body1" color="text.secondary">
                  No sentiment data available. Start streaming to see real-time analysis.
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Sentiment Timeline */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            {sentimentTimelineData ? (
              <Line data={sentimentTimelineData} options={timelineOptions} />
            ) : (
              <Box sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Real-time Sentiment Trends
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Timeline will appear once streaming data is available.
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Recent Tweets */}
        <Grid item xs={12}>
          <TweetList
            tweets={streamingData?.tweets?.slice(-20) || []}
            title="Recent Streaming Tweets"
          />
        </Grid>
      </Grid>
    </Container>
  );
};

export default RealTimeDashboard;