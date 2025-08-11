// frontend/src/pages/JobDetailPage.jsx

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom'; // useParams to get ID from URL, Link to go back
import { useAuth } from '../context/AuthContext.jsx';
import { getJobById, getDownloadUrl } from '../services/api';
import './JobDetailPage.css'; // We will create this new CSS file next

// Helper to format dates
const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
};

// Helper to render the status with a spinner and text
const renderStatus = (status) => {
    switch (status) {
        case 'PENDING':
            return <><div className="spinner"></div><span>Queued</span></>;
        case 'PROGRESS':
            return <><div className="spinner"></div><span>Editing...</span></>;
        case 'SUCCESS':
            return <span className="status-success">✔ Complete</span>;
        case 'FAILURE':
            return <span className="status-failure">✖ Failed</span>;
        default:
            return <span>{status}</span>;
    }
};

function JobDetailPage() {
    const { jobId } = useParams(); // Get the job ID from the URL, e.g., "/jobs/abc-123"
    const { token } = useAuth();

    const [job, setJob] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [followUpPrompt, setFollowUpPrompt] = useState('');

    // This function fetches the latest job data. It's wrapped in useCallback
    // to be stable and not cause unnecessary re-renders when used in useEffect.
    const fetchJob = useCallback(async () => {
        if (!token || !jobId) return;
        try {
            // NOTE: The getJobById function currently uses the `/status` endpoint.
            // This endpoint returns the core job data but might be missing metadata
            // like the original prompt. For now, we merge it with existing data.
            // A dedicated `/jobs/{jobId}` endpoint on the backend would be ideal.
            const fetchedJobData = await getJobById(jobId, token);
            setJob(prevJob => ({
                ...prevJob, // Keep old data like prompt if it's not in the new payload
                ...fetchedJobData,
                result_payload: fetchedJobData.result, // Align payload key
            }));
        } catch (err) {
            console.error(`Failed to fetch job ${jobId}:`, err);
            setError(err.message || 'An unknown error occurred.');
        }
    }, [jobId, token]);

    // Effect for the initial data load when the component mounts.
    useEffect(() => {
        setIsLoading(true);
        fetchJob().finally(() => setIsLoading(false));
    }, [fetchJob]); // Runs once on mount because fetchJob is stable

    // Effect for polling, which starts/stops based on the job's status.
    useEffect(() => {
        if (!job || (job.status !== 'PENDING' && job.status !== 'PROGRESS')) {
            return; // Stop polling if the job is done or doesn't exist.
        }

        const intervalId = setInterval(fetchJob, 5000); // Poll every 5 seconds

        // Cleanup function to stop polling when the component unmounts or the job is finished.
        return () => clearInterval(intervalId);
    }, [job, fetchJob]);

    const handleDownloadClick = (e) => {
        e.preventDefault();
        fetch(getDownloadUrl(job.job_id), { headers: { 'Authorization': `Bearer ${token}` } })
            .then(res => {
                if (!res.ok) throw new Error(`Download failed: ${res.statusText}`);
                return res.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                const filename = job.result_payload?.output_path?.split('/').pop();
                a.download = filename || 'codec_edit.otio';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
            })
            .catch(err => {
                console.error("Download error:", err);
                setError("Download failed. Please try again.");
            });
    };

    const handleFollowUpSubmit = (e) => {
        e.preventDefault();
        if (!followUpPrompt.trim()) return;
        // --- FUTURE ---
        // This is where you would call a new API endpoint, e.g.,
        // `askFollowUp(jobId, followUpPrompt, token)`
        console.log(`Submitting follow-up for job ${jobId}: "${followUpPrompt}"`);
        alert("Follow-up functionality is not yet implemented in the backend.");
        setFollowUpPrompt('');
    };

    if (isLoading) {
        return <div className="loading-container"><h2>Loading Job Details...</h2></div>;
    }

    if (error) {
        return (
            <div className="job-detail-page">
                <header className="job-detail-header">
                    <Link to="/" className="back-link">&larr; Back to Dashboard</Link>
                </header>
                <div className="error-text">Error: {error}</div>
            </div>
        );
    }

    if (!job) {
        return (
            <div className="job-detail-page">
                <header className="job-detail-header">
                    <Link to="/" className="back-link">&larr; Back to Dashboard</Link>
                </header>
                <p>Job not found.</p>
            </div>
        );
    }

    return (
        <div className="job-detail-page">
            <header className="job-detail-header">
                <Link to="/" className="back-link">&larr; Back to Dashboard</Link>
                <div className="job-status-header">{renderStatus(job.status)}</div>
            </header>

            <main className="job-detail-content">
                <div className="job-prompt-section">
                    <h3>Original Prompt</h3>
                    <p>"{job.prompt || 'Could not load prompt. This may be an in-progress job.'}"</p>
                    <span className="job-meta">Job ID: {job.job_id} &bull; Created: {formatDate(job.created_at)}</span>
                </div>

                <div className="job-conversation-section">
                    <h3>Agent Log</h3>
                    <div className="agent-message-box">
                        <pre>{job.result_payload?.message || 'Waiting for agent to start...'}</pre>
                    </div>
                </div>

                {job.status === 'SUCCESS' && job.result_payload?.output_path && (
                    <div className="job-actions-section">
                        <h3>Result</h3>
                        <button onClick={handleDownloadClick} className="download-button-large">
                            Download Edit
                        </button>
                    </div>
                )}

                {job.status === 'FAILURE' && (
                     <div className="job-actions-section">
                        <h3>Result</h3>
                        <p className="error-text">This job failed to complete.</p>
                    </div>
                )}

                <hr className="divider" />

                <div className="follow-up-section">
                    <h3>Ask a Follow-up</h3>
                    <form onSubmit={handleFollowUpSubmit}>
                        <textarea
                            value={followUpPrompt}
                            onChange={(e) => setFollowUpPrompt(e.target.value)}
                            placeholder="e.g., 'Make it shorter.' or 'Change the background music.'"
                            rows="3"
                            disabled={job.status !== 'SUCCESS'} // Only allow follow-ups on success
                        />
                        <button type="submit" disabled={!followUpPrompt.trim() || job.status !== 'SUCCESS'}>
                            Send
                        </button>
                    </form>
                </div>
            </main>
        </div>
    );
}

export default JobDetailPage;