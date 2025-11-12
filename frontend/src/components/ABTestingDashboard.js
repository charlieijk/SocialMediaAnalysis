import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Alert,
} from '@mui/material';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const ABTestingDashboard = () => {
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [experimentReport, setExperimentReport] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Create experiment dialog state
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    description: '',
    models_to_test: []
  });

  useEffect(() => {
    fetchExperiments();
    fetchAvailableModels();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/ab-testing/experiments`);
      setExperiments(response.data.experiments);
    } catch (err) {
      setError('Failed to fetch experiments');
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/ab-testing/models/available`);
      setAvailableModels(response.data.models);
    } catch (err) {
      setError('Failed to fetch available models');
    }
  };

  const createExperiment = async () => {
    if (!newExperiment.name || newExperiment.models_to_test.length === 0) {
      setError('Name and at least one model are required');
      return;
    }

    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/api/ab-testing/experiments`, newExperiment);
      setSuccess('Experiment created successfully');
      setCreateDialogOpen(false);
      setNewExperiment({ name: '', description: '', models_to_test: [] });
      fetchExperiments();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to create experiment');
    } finally {
      setLoading(false);
    }
  };

  const runQuickTest = async (experimentId) => {
    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/quick-test`);
      setSuccess('Quick test completed successfully');
      fetchExperiments();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to run quick test');
    } finally {
      setLoading(false);
    }
  };

  const fetchExperimentReport = async (experimentId) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/report`);
      setExperimentReport(response.data);
      setSelectedExperiment(experiments.find(exp => exp.id === experimentId));
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch experiment report');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'warning';
      case 'created': return 'default';
      default: return 'default';
    }
  };

  const getMetricsChartData = () => {
    if (!experimentReport || !experimentReport.detailed_metrics) return null;

    const models = Object.keys(experimentReport.detailed_metrics).filter(key =>
      key !== 'speed' && experimentReport.detailed_metrics[key].values
    );

    if (models.length === 0) return null;

    const metrics = Object.keys(experimentReport.detailed_metrics).filter(key => key !== 'speed');
    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];

    const datasets = metrics.map((metric, index) => ({
      label: metric.charAt(0).toUpperCase() + metric.slice(1),
      data: models.map(model =>
        experimentReport.detailed_metrics[metric]?.values?.[model] || 0
      ),
      backgroundColor: colors[index % colors.length],
      borderColor: colors[index % colors.length],
      borderWidth: 1,
    }));

    return {
      labels: models,
      datasets: datasets
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        A/B Testing Dashboard
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

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Controls */}
      <Box sx={{ mb: 3 }}>
        <Button
          variant="contained"
          onClick={() => setCreateDialogOpen(true)}
          sx={{ mr: 2 }}
        >
          Create New Experiment
        </Button>
        <Button
          variant="outlined"
          onClick={fetchExperiments}
        >
          Refresh
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Experiments List */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Experiments
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Models</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {experiments.map((experiment) => (
                    <TableRow key={experiment.id}>
                      <TableCell>{experiment.name}</TableCell>
                      <TableCell>
                        <Chip
                          label={experiment.status}
                          color={getStatusColor(experiment.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {experiment.models_tested?.length || 0} models
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {experiment.status === 'created' && (
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => runQuickTest(experiment.id)}
                            >
                              Quick Test
                            </Button>
                          )}
                          {experiment.status === 'completed' && (
                            <Button
                              size="small"
                              variant="contained"
                              onClick={() => fetchExperimentReport(experiment.id)}
                            >
                              View Report
                            </Button>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            {experiments.length === 0 && (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  No experiments found. Create your first experiment to get started.
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Experiment Results */}
        <Grid item xs={12} md={6}>
          {experimentReport ? (
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Experiment Results: {selectedExperiment?.name}
              </Typography>

              {/* Model Rankings */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Model Rankings (Overall)
                </Typography>
                {experimentReport.model_rankings?.overall?.map(([rank, model, score]) => (
                  <Box key={model} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">
                      {rank}. {model}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {(score * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                ))}
              </Box>

              {/* Best Models by Metric */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Best Models by Metric
                </Typography>
                <Grid container spacing={1}>
                  {Object.entries(experimentReport.detailed_metrics || {})
                    .filter(([key]) => key !== 'speed')
                    .map(([metric, data]) => (
                      <Grid item xs={6} key={metric}>
                        <Card variant="outlined" sx={{ p: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            {metric.charAt(0).toUpperCase() + metric.slice(1)}
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {data.best_model}
                          </Typography>
                        </Card>
                      </Grid>
                    ))}
                </Grid>
              </Box>

              {/* Recommendations */}
              {experimentReport.recommendations && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Recommendations
                  </Typography>
                  {experimentReport.recommendations.map((rec, index) => (
                    <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                      • {rec}
                    </Typography>
                  ))}
                </Box>
              )}
            </Paper>
          ) : (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="body1" color="text.secondary">
                Select an experiment to view detailed results
              </Typography>
            </Paper>
          )}
        </Grid>

        {/* Performance Chart */}
        {experimentReport && getMetricsChartData() && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Bar data={getMetricsChartData()} options={chartOptions} />
            </Paper>
          </Grid>
        )}
      </Grid>

      {/* Create Experiment Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New A/B Test Experiment</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Experiment Name"
            value={newExperiment.name}
            onChange={(e) => setNewExperiment({...newExperiment, name: e.target.value})}
            sx={{ mb: 2, mt: 1 }}
          />
          <TextField
            fullWidth
            label="Description"
            multiline
            rows={3}
            value={newExperiment.description}
            onChange={(e) => setNewExperiment({...newExperiment, description: e.target.value})}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth>
            <InputLabel>Models to Test</InputLabel>
            <Select
              multiple
              value={newExperiment.models_to_test}
              onChange={(e) => setNewExperiment({...newExperiment, models_to_test: e.target.value})}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selected.map((value) => (
                    <Chip key={value} label={value} size="small" />
                  ))}
                </Box>
              )}
            >
              {availableModels.map((model) => (
                <MenuItem key={model} value={model}>
                  {model}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={createExperiment} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ABTestingDashboard;