// frontend/src/pages/JobDetailPage.jsx

import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom'; // useParams to get ID from URL, Link is removed
import { useAuth } from '../context/AuthContext.jsx';
import { getJobById, getDownloadUrl } from '../services/api';
import './JobDetailPage.css'; // This CSS file will be updated next

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
    const { jobId } = useParams();
    const { token } = useAuth();

    const [job, setJob] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [followUpPrompt, setFollowUpPrompt] = useState('');

    const fetchJob = useCallback(async () => {
        if (!token || !jobId) return;
        try {
            const fetchedJobData = await getJobById(jobId, token);
            setJob(prevJob => ({
                ...prevJob,
                ...fetchedJobData,
                result_payload: fetchedJobData.result,
            }));
        } catch (err) {
            console.error(`Failed to fetch job ${jobId}:`, err);
            setError(err.message || 'An unknown error occurred.');
        }
    }, [jobId, token]);

    // Effect for the initial data load (no changes)
    useEffect(() => {
        setIsLoading(true);
        fetchJob().finally(() => setIsLoading(false));
    }, [fetchJob]);

    // Effect for polling (no changes)
    useEffect(() => {
        if (!job || (job.status !== 'PENDING' && job.status !== 'PROGRESS')) {
            return;
        }
        const intervalId = setInterval(fetchJob, 5000);
        return () => clearInterval(intervalId);
    }, [job, fetchJob]);

    // Download handler (no changes)
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

    // Follow-up handler (no changes)
    const handleFollowUpSubmit = (e) => {
        e.preventDefault();
        if (!followUpPrompt.trim()) return;
        console.log(`Submitting follow-up for job ${jobId}: "${followUpPrompt}"`);
        alert("Follow-up functionality is not yet implemented in the backend.");
        setFollowUpPrompt('');
    };

    if (isLoading) {
        return <div className="loading-container"><h2>Loading Job Details...</h2></div>;
    }

    // Updated error/not-found views to remove the back link
    if (error) {
        return (
            <div className="job-detail-page">
                <div className="error-text">Error: {error}</div>
            </div>
        );
    }

    if (!job) {
        return (
            <div className="job-detail-page">
                <p>Job not found.</p>
            </div>
        );
    }

    return (
        <div className="job-detail-page">
            {/* The header is simplified, removing the back link. */}
            <header className="job-detail-header">
                <h1>Edit Details</h1>
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
                            disabled={job.status !== 'SUCCESS'}
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