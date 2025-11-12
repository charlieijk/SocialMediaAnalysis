import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const searchTweets = async (query, maxResults = 100) => {
  try {
    const response = await api.post('/api/search', {
      query,
      max_results: maxResults,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to search tweets');
  }
};

export const startStreaming = async (keywords) => {
  try {
    const response = await api.post('/api/stream/start', {
      keywords,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to start streaming');
  }
};

export const stopStreaming = async () => {
  try {
    const response = await api.post('/api/stream/stop');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to stop streaming');
  }
};

export const getStreamingData = async () => {
  try {
    const response = await api.get('/api/stream/data');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to get streaming data');
  }
};

export const analyzeSentiment = async (texts, model = 'ensemble') => {
  try {
    const response = await api.post('/api/sentiment/analyze', {
      texts,
      model,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to analyze sentiment');
  }
};

export const getTrendingTopics = async () => {
  try {
    const response = await api.get('/api/trends');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to get trending topics');
  }
};

export const getUserTimeline = async (username, maxResults = 100) => {
  try {
    const response = await api.get(`/api/user/${username}?max_results=${maxResults}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to get user timeline');
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get('/api/health');
    return response.data;
  } catch (error) {
    throw new Error('API is not available');
  }
};

export default api;