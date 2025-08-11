// frontend/src/components/Sidebar.jsx

import React, { useState, useEffect, useCallback } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import { getJobs } from '../services/api';
import './Sidebar.css';

function Sidebar({ isVisible }) {
    const { token } = useAuth();
    const [jobs, setJobs] = useState([]);
    const [error, setError] = useState(null);
    // We don't need a separate loading state, as the list will just be empty initially.

    const fetchJobs = useCallback(async () => {
        if (!token) return;
        try {
            const fetchedJobs = await getJobs(token);
            // Sort jobs to show active ones first, then by creation date
            fetchedJobs.sort((a, b) => {
                const aIsActive = a.status === 'PENDING' || a.status === 'PROGRESS';
                const bIsActive = b.status === 'PENDING' || b.status === 'PROGRESS';
                if (aIsActive && !bIsActive) return -1;
                if (!aIsActive && bIsActive) return 1;
                return new Date(b.created_at) - new Date(a.created_at);
            });
            setJobs(fetchedJobs);
            setError(null); // Clear previous errors on successful fetch
        } catch (err) {
            console.error("Error fetching jobs for sidebar:", err);
            setError('Could not load edits.');
        }
    }, [token]);

    // Fetch jobs on mount and then poll for updates every 5 seconds.
    // This keeps the status indicators (like the spinner) up-to-date.
    useEffect(() => {
        fetchJobs(); // Initial fetch
        const intervalId = setInterval(fetchJobs, 5000); // Poll every 5 seconds

        // Cleanup interval on component unmount
        return () => clearInterval(intervalId);
    }, [fetchJobs]);

    const renderJobItem = (job) => {
        const isActive = job.status === 'PENDING' || job.status === 'PROGRESS';
        return (
            <li key={job.job_id}>
                <NavLink
                    to={`/jobs/${job.job_id}`}
                    className={({ isActive }) => `job-link ${isActive ? 'active' : ''}`}
                >
                    <span className="job-link-prompt">
                        {job.prompt || 'Untitled Edit'}
                    </span>
                    {isActive && <div className="spinner-small"></div>}
                </NavLink>
            </li>
        );
    };

    return (
        <aside className={`sidebar ${isVisible ? 'visible' : ''}`}>
            <div className="sidebar-header">
                <Link to="/" className="new-edit-button">
                    + New Edit
                </Link>
            </div>
            <nav className="job-list-nav">
                {error && <p className="sidebar-error">{error}</p>}
                {jobs.length > 0 ? (
                    <ul>
                        {jobs.map(renderJobItem)}
                    </ul>
                ) : (
                    !error && <p className="sidebar-info">No past edits found.</p>
                )}
            </nav>
        </aside>
    );
}

export default Sidebar;