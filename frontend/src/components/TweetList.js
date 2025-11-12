import React from 'react';
import {
  List,
  ListItem,
  ListItemText,
  Paper,
  Chip,
  Typography,
  Box,
  Avatar,
} from '@mui/material';
import {
  SentimentVeryDissatisfied,
  SentimentNeutral,
  SentimentVerySatisfied,
} from '@mui/icons-material';

const TweetList = ({ tweets, title = "Recent Tweets" }) => {
  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return <SentimentVerySatisfied sx={{ color: '#4CAF50' }} />;
      case 'negative':
        return <SentimentVeryDissatisfied sx={{ color: '#F44336' }} />;
      default:
        return <SentimentNeutral sx={{ color: '#FFC107' }} />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return '#4CAF50';
      case 'negative':
        return '#F44336';
      default:
        return '#FFC107';
    }
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return 'Unknown time';
    }
  };

  if (!tweets || tweets.length === 0) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          No tweets available
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ maxHeight: 600, overflow: 'auto' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">
          {title} ({tweets.length})
        </Typography>
      </Box>
      <List>
        {tweets.map((tweet) => (
          <ListItem key={tweet.id} divider>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', width: '100%' }}>
              <Avatar sx={{ mr: 2, mt: 1 }}>
                {getSentimentIcon(tweet.sentiment?.label)}
              </Avatar>
              <Box sx={{ flexGrow: 1 }}>
                <ListItemText
                  primary={
                    <Typography variant="body1" sx={{ mb: 1 }}>
                      {tweet.text}
                    </Typography>
                  }
                  secondary={
                    <Box>
                      <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                        <Chip
                          label={`${tweet.sentiment?.label || 'neutral'} (${(tweet.sentiment?.confidence * 100).toFixed(1)}%)`}
                          size="small"
                          sx={{
                            backgroundColor: getSentimentColor(tweet.sentiment?.label),
                            color: 'white',
                          }}
                        />
                        {tweet.public_metrics && (
                          <>
                            <Chip
                              label={`❤️ ${tweet.public_metrics.like_count || 0}`}
                              size="small"
                              variant="outlined"
                            />
                            <Chip
                              label={`🔄 ${tweet.public_metrics.retweet_count || 0}`}
                              size="small"
                              variant="outlined"
                            />
                            <Chip
                              label={`💬 ${tweet.public_metrics.reply_count || 0}`}
                              size="small"
                              variant="outlined"
                            />
                          </>
                        )}
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(tweet.created_at)} • ID: {tweet.id}
                      </Typography>
                    </Box>
                  }
                />
              </Box>
            </Box>
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default TweetList;