// frontend/src/App.jsx

import React from 'react';
// NEW: Import routing components
import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import LoginPage from './pages/LoginPage.jsx';
import MainPage from './pages/MainPage.jsx';
// NEW: Import the (yet to be created) JobDetailPage
import JobDetailPage from './pages/JobDetailPage.jsx';
import './App.css';

/**
 * This is the core content of our app.
 * It's a separate component so it can be a child of AuthProvider
 * and use the `useAuth` hook to manage routing.
 */
function AppContent() {
  const { isLoggedIn, isLoading } = useAuth();

  // While the AuthContext is checking for a stored token,
  // we show a simple loading screen.
  if (isLoading) {
    return (
      <div className="loading-container">
        <h2>Loading Codec...</h2>
      </div>
    );
  }

  // Once loading is complete, we define the routes.
  return (
    <Routes>
      {isLoggedIn ? (
        // --- Routes for Authenticated Users ---
        <>
          {/* The root path '/' now renders our main dashboard (MainPage) */}
          <Route path="/" element={<MainPage />} />

          {/* The new detail page for a specific job */}
          <Route path="/jobs/:jobId" element={<JobDetailPage />} />

          {/* A catch-all route for logged-in users. If they go to a
              non-existent URL (like /foo), redirect them to the dashboard. */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </>
      ) : (
        // --- Routes for Unauthenticated Users ---
        <>
          {/* If not logged in, any path will render the LoginPage.
              This also handles the case where a logged-out user tries
              to access a protected URL directly. */}
          <Route path="*" element={<LoginPage />} />
        </>
      )}
    </Routes>
  );
}

/**
 * The main App component. Its only job is to provide the AuthContext
 * and the main layout to the rest of the application.
 */
function App() {
  return (
    <AuthProvider>
      <div className="App">
        <AppContent />
      </div>
    </AuthProvider>
  );
}

export default App;