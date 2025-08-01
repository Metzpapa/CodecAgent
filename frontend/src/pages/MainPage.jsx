// frontend/src/pages/MainPage.js

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import { getJobs } from '../services/api';
import CreateJobForm from '../components/CreateJobForm.jsx';
import JobList from '../components/JobList.jsx';
import './MainPage.css';

function MainPage() {
    const { user, token, logout } = useAuth();
    const [jobs, setJobs] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    // We wrap fetchJobs in useCallback to memoize it.
    // This prevents it from being recreated on every render, which is efficient.
    const fetchJobs = useCallback(async () => {
        if (!token) return;

        setIsLoading(true);
        setError(null);
        try {
            const fetchedJobs = await getJobs(token);
            setJobs(fetchedJobs);
        } catch (err) {
            console.error("Error fetching jobs:", err);
            setError(err.message || 'An unknown error occurred while fetching jobs.');
        } finally {
            setIsLoading(false);
        }
    }, [token]); // It only needs to be recreated if the token changes.

    // Fetch jobs when the component mounts.
    useEffect(() => {
        fetchJobs();
    }, [fetchJobs]);

    // This function will be passed down to the CreateJobForm.
    // When a new job is successfully created, the form will call this
    // function to trigger a refresh of the job list.
    const handleJobCreated = () => {
        console.log("New job created. Refreshing job list...");
        fetchJobs();
    };

    return (
        <div className="main-page-container">
            <header className="main-header">
                <div className="logo">Codec</div>
                <div className="user-info">
                    <span>{user?.email}</span>
                    <button onClick={logout} className="logout-button">Logout</button>
                </div>
            </header>

            <main className="main-content">
                <CreateJobForm onJobCreated={handleJobCreated} />
                
                <hr className="divider" />

                <JobList jobs={jobs} isLoading={isLoading} error={error} />
            </main>
        </div>
    );
}

export default MainPage;