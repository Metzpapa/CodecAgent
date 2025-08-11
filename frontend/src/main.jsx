// frontend/src/main.jsx

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom'; // <-- Import BrowserRouter
import App from './App.jsx'; // Your main App component

// This finds the <div id="root"> in your index.html
const rootElement = document.getElementById('root');

// This creates the React root and renders your App component into it.
ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    {/* Wrap the entire App component with BrowserRouter */}
    {/* This makes routing available to all child components */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);