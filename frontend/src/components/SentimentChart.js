import React from 'react';
import { Pie, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const SentimentChart = ({ sentimentStats, chartType = 'pie' }) => {
  if (!sentimentStats) {
    return <div>No sentiment data available</div>;
  }

  const { counts, percentages } = sentimentStats;

  const pieData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: [counts.positive, counts.neutral, counts.negative],
        backgroundColor: [
          '#4CAF50', // Green for positive
          '#FFC107', // Yellow for neutral
          '#F44336', // Red for negative
        ],
        borderColor: [
          '#45a049',
          '#e6ac00',
          '#d32f2f',
        ],
        borderWidth: 2,
      },
    ],
  };

  const barData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        label: 'Sentiment Distribution',
        data: [percentages.positive, percentages.neutral, percentages.negative],
        backgroundColor: [
          'rgba(76, 175, 80, 0.6)',
          'rgba(255, 193, 7, 0.6)',
          'rgba(244, 67, 54, 0.6)',
        ],
        borderColor: [
          'rgba(76, 175, 80, 1)',
          'rgba(255, 193, 7, 1)',
          'rgba(244, 67, 54, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const pieOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
      },
      title: {
        display: true,
        text: 'Sentiment Distribution',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed;
            const percentage = percentages[label.toLowerCase()];
            return `${label}: ${value} (${percentage.toFixed(1)}%)`;
          },
        },
      },
    },
  };

  const barOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Sentiment Distribution (%)',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%';
          },
        },
      },
    },
  };

  return (
    <div style={{ width: '100%', height: '400px' }}>
      {chartType === 'pie' ? (
        <Pie data={pieData} options={pieOptions} />
      ) : (
        <Bar data={barData} options={barOptions} />
      )}
    </div>
  );
};

export default SentimentChart;