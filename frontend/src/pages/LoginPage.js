// frontend/src/pages/LoginPage.js

import React, { useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import './LoginPage.css'; // We'll create this for styling

function LoginPage() {
    // Get the function to render the Google button from our AuthContext.
    const { renderLoginButton } = useAuth();

    // We use useEffect to ensure that the Google button is rendered only after
    // the component has mounted and the 'google-signin-button' div is available in the DOM.
    useEffect(() => {
        // The ID 'google-signin-button' must match the ID of the div below.
        renderLoginButton('google-signin-button');
    }, [renderLoginButton]); // The dependency array ensures this runs only when the function is available.

    return (
        <div className="login-page-container">
            <div className="login-box">
                <header className="login-header">
                    <h1>Codec</h1>
                    <p>AI-Powered Video Editing</p>
                </header>
                <main className="login-main">
                    <p className="login-instructions">
                        Sign in to start your next edit.
                    </p>
                    {/* 
                      This div is the target for the Google Sign-In button.
                      The Google Identity Services script will replace this div
                      with the fully functional sign-in button.
                    */}
                    <div id="google-signin-button"></div>
                </main>
            </div>
        </div>
    );
}

export default LoginPage;