import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Tabs,
  Tab,
  Box,
} from '@mui/material';
import { Analytics, Search, TrendingUp, Science } from '@mui/icons-material';
import RealTimeDashboard from './components/RealTimeDashboard';
import SearchPage from './pages/SearchPage';
import ABTestingDashboard from './components/ABTestingDashboard';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static">
            <Toolbar>
              <Analytics sx={{ mr: 2 }} />
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Social Media Sentiment Analysis
              </Typography>
            </Toolbar>
          </AppBar>

          <Container maxWidth={false} disableGutters>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                aria-label="navigation tabs"
                centered
              >
                <Tab
                  icon={<TrendingUp />}
                  label="Real-time Dashboard"
                  iconPosition="start"
                />
                <Tab
                  icon={<Search />}
                  label="Search & Analyze"
                  iconPosition="start"
                />
                <Tab
                  icon={<Science />}
                  label="A/B Testing"
                  iconPosition="start"
                />
              </Tabs>
            </Box>

            <TabPanel value={tabValue} index={0}>
              <RealTimeDashboard />
            </TabPanel>
            <TabPanel value={tabValue} index={1}>
              <SearchPage />
            </TabPanel>
            <TabPanel value={tabValue} index={2}>
              <ABTestingDashboard />
            </TabPanel>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;