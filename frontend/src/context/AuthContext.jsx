// frontend/src/context/AuthContext.jsx

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
            // On any successful login (manual or silent), we are done loading.
            setIsLoading(false);
        };

        // This function contains all the logic that depends on `window.google`.
        // It will only be called after we are sure the GSI script has loaded.
        const initializeGoogleSignIn = () => {
            try {
                // Initialize the Google Sign-In client
                window.google.accounts.id.initialize({
                    client_id: googleClientId,
                    callback: handleCredentialResponse,
                });

                const storedToken = localStorage.getItem('googleIdToken');
                if (storedToken) {
                    const decodedToken = jwtDecode(storedToken);
                    // Check if the token is expired
                    if (decodedToken.exp * 1000 > Date.now()) {
                        // Token is valid and not expired. We're logged in.
                        setUser(decodedToken);
                        setToken(storedToken);
                        setIsLoading(false); // Stop loading, we're good to go.
                    } else {
                        // Token is expired. Attempt a silent sign-in.
                        console.log("Stored token is expired. Attempting silent sign-in...");
                        localStorage.removeItem('googleIdToken');

                        // This triggers the silent sign-in / One-Tap prompt.
                        // If successful, `handleCredentialResponse` will be called.
                        // If it fails or is dismissed, the notification callback handles it.
                        window.google.accounts.id.prompt((notification) => {
                            // This callback is key for knowing when the prompt flow fails.
                            const isFailure = notification.isNotDisplayed() || notification.isSkippedMoment() || notification.isDismissedMoment();
                            if (isFailure) {
                                console.log("Silent/One-Tap sign-in did not result in a credential. User needs to log in manually.");
                                // We can now safely stop loading and show the login page.
                                setIsLoading(false);
                            }
                        });
                        // We keep `isLoading` as `true` while we wait for the prompt's outcome.
                    }
                } else {
                    // No token exists. User is not logged in. Stop loading.
                    setIsLoading(false);
                }
            } catch (error) {
                console.error("Error initializing Google Sign-In or decoding token:", error);
                // Clear potentially corrupt token
                localStorage.removeItem('googleIdToken');
                setIsLoading(false);
            }
        };

        // Check if the Google script is already loaded (e.g., from cache on reload)
        if (window.google) {
            console.log("Google script already loaded. Initializing...");
            initializeGoogleSignIn();
        } else {
            // If not, wait for the entire window to load. The `defer` attribute on the
            // script tag in index.html ensures it will run before the `load` event.
            console.log("Google script not loaded yet. Waiting for window.onload...");
            window.addEventListener('load', initializeGoogleSignIn);

            // Cleanup the event listener when the component unmounts
            return () => {
                window.removeEventListener('load', initializeGoogleSignIn);
            };
        }

    }, []); // This effect should still only run once on mount.

    // Function to render the Google Sign-In button
    const renderLoginButton = (elementId) => {
        // This function might be called by a component before the GSI script is ready.
        // We add a small delay and retry to make it more robust.
        const attemptRender = () => {
            if (window.google && window.google.accounts && window.google.accounts.id) {
                window.google.accounts.id.renderButton(
                    document.getElementById(elementId),
                    { theme: "outline", size: "large" } // Customize the button's appearance
                );
            } else {
                console.error("Google Identity Services is not available.");
            }
        };

        if (window.google) {
            attemptRender();
        } else {
            // If the script isn't loaded at all yet, wait a moment and try again.
            setTimeout(attemptRender, 500);
        }
    };

    // Function to handle user logout
    const logout = () => {
        setUser(null);
        setToken(null);
        localStorage.removeItem('googleIdToken');
        // Optional: Revoke the token on Google's side
        if (token && window.google) {
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