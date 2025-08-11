// frontend/src/App.jsx

import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import LoginPage from './pages/LoginPage.jsx';
import MainPage from './pages/MainPage.jsx';
import JobDetailPage from './pages/JobDetailPage.jsx';
import AppLayout from './components/AppLayout.jsx'; // <-- NEW: Import the layout
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
        // --- NEW: Routes for Authenticated Users ---
        // All authenticated routes are now children of the AppLayout route.
        // This means AppLayout will always be rendered, and the child route's
        // element will be rendered in AppLayout's <Outlet />.
        <Route path="/" element={<AppLayout />}>
          {/* The `index` route is the default child, shown at the parent's path ('/') */}
          <Route index element={<MainPage />} />

          {/* The detail page for a specific job */}
          <Route path="jobs/:jobId" element={<JobDetailPage />} />

          {/* A catch-all route for logged-in users. If they go to a
              non-existent URL (like /foo), redirect them to the dashboard. */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      ) : (
        // --- Routes for Unauthenticated Users ---
        // If not logged in, any path will render the LoginPage.
        <Route path="*" element={<LoginPage />} />
      )}
    </Routes>
  );
}

/**
 * The main App component. Its only job is to provide the AuthContext.
 * The main div has been removed as AppLayout now controls the entire viewport.
 */
function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;