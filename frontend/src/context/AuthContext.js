// frontend/src/context/AuthContext.js

import React, { createContext, useState, useEffect, useContext } from 'react';
import { jwtDecode } from 'jwt-decode';

// Create the context with a default value
const AuthContext = createContext(null);

// A custom hook to make it easier to use the auth context in other components
export const useAuth = () => {
    return useContext(AuthContext);
};

// The provider component that will wrap our entire application
export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null); // Will store decoded user info (name, email, etc.)
    const [token, setToken] = useState(null); // Will store the raw Google ID token (JWT)
    const [isLoading, setIsLoading] = useState(true); // To handle initial auth state loading

    // This effect runs once on component mount to initialize Google Identity Services
    useEffect(() => {
        const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
        if (!googleClientId) {
            console.error("VITE_GOOGLE_CLIENT_ID is not set in .env file.");
            setIsLoading(false);
            return;
        }

        // The callback function that Google will call after a user signs in
        const handleCredentialResponse = (response) => {
            console.log("Encoded JWT ID token: " + response.credential);
            const idToken = response.credential;
            const decodedToken = jwtDecode(idToken);

            console.log("Decoded User Info:", decodedToken);
            setUser(decodedToken);
            setToken(idToken);
            // Store the token in localStorage to persist the session across page reloads
            localStorage.setItem('googleIdToken', idToken);
        };

        // Initialize the Google Sign-In client
        window.google.accounts.id.initialize({
            client_id: googleClientId,
            callback: handleCredentialResponse,
        });

        // Check if a token already exists in localStorage from a previous session
        const storedToken = localStorage.getItem('googleIdToken');
        if (storedToken) {
            try {
                const decodedToken = jwtDecode(storedToken);
                // Check if the token is expired
                if (decodedToken.exp * 1000 > Date.now()) {
                    setUser(decodedToken);
                    setToken(storedToken);
                } else {
                    // Token is expired, clear it
                    localStorage.removeItem('googleIdToken');
                }
            } catch (error) {
                console.error("Error decoding stored token:", error);
                localStorage.removeItem('googleIdToken');
            }
        }
        setIsLoading(false); // Finished loading auth state

    }, []);

    // Function to render the Google Sign-In button
    const renderLoginButton = (elementId) => {
        if (window.google && window.google.accounts && window.google.accounts.id) {
            window.google.accounts.id.renderButton(
                document.getElementById(elementId),
                { theme: "outline", size: "large" } // Customize the button's appearance
            );
        } else {
            console.error("Google Identity Services is not available.");
        }
    };

    // Function to handle user logout
    const logout = () => {
        setUser(null);
        setToken(null);
        localStorage.removeItem('googleIdToken');
        // Optional: Revoke the token on Google's side
        if (token) {
            window.google.accounts.id.revoke(user.email, done => {
                console.log('User token revoked.');
            });
        }
        console.log("User logged out.");
    };

    // The value that will be provided to all consuming components
    const value = {
        user,
        token,
        isLoggedIn: !!user, // A convenient boolean flag
        isLoading,
        renderLoginButton,
        logout,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};