// frontend/src/main.jsx

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx'; // Your main App component

// This finds the <div id="root"> in your index.html
const rootElement = document.getElementById('root');

// This creates the React root and renders your App component into it.
ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);