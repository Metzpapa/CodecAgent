// frontend/src/App.js

import React from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import MainPage from './pages/MainPage';
import './App.css'; // We'll create this for basic styling

/**
 * This is the core content of our app.
 * It's a separate component so it can be a child of AuthProvider
 * and use the `useAuth` hook.
 */
function AppContent() {
  const { isLoggedIn, isLoading } = useAuth();

  // While the AuthContext is checking for a stored token,
  // we show a simple loading screen. This prevents a "flash"
  // of the login page if the user is already authenticated.
  if (isLoading) {
    return (
      <div className="loading-container">
        <h2>Loading Codec...</h2>
      </div>
    );
  }

  // Once loading is complete, we render the correct page.
  return (
    <>
      {isLoggedIn ? <MainPage /> : <LoginPage />}
    </>
  );
}

/**
 * The main App component. Its only job is to provide the AuthContext
 * to the rest of the application.
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