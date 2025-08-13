// frontend/src/pages/JobDetailPage.jsx

import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import { getJobById, getDownloadUrl } from '../services/api';
import './JobDetailPage.css';

// Helper to format dates
const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
};

// Helper to get a simple status text for the card header
const getStatusText = (status) => {
    switch (status) {
        case 'PENDING': return 'Queued';
        case 'PROGRESS': return 'In Progress';
        case 'SUCCESS': return 'Complete';
        case 'FAILURE': return 'Failed';
        default: return status || 'Loading...';
    }
};

function JobDetailPage() {
    const { jobId } = useParams();
    // --- FIX: Destructure `isLoading` from auth and rename it to avoid conflicts ---
    const { token, isLoading: isAuthLoading } = useAuth();

    const [job, setJob] = useState(null);
    // Renamed to be more specific
    const [isPageLoading, setIsPageLoading] = useState(true);
    const [error, setError] = useState(null);
    const [followUpPrompt, setFollowUpPrompt] = useState('');

    // --- FIX: This effect now handles initial load and resets state correctly ---
    useEffect(() => {
        // Don't do anything if auth is still resolving or if we have no token.
        if (isAuthLoading || !token) {
            return;
        }

        const loadInitialJob = async () => {
            setError(null); // <-- FIX for sticky error: Reset on every new job load.
            setIsPageLoading(true);
            try {
                const fetchedJobData = await getJobById(jobId, token);
                // The backend returns the payload in a 'result' key. We map it to
                // 'result_payload' which the rest of this component expects.
                setJob({ ...fetchedJobData, result_payload: fetchedJobData.result });
            } catch (err) {
                console.error(`Failed to fetch job ${jobId}:`, err);
                setError(err.message || 'An unknown error occurred.');
            } finally {
                setIsPageLoading(false);
            }
        };

        loadInitialJob();
    }, [jobId, token, isAuthLoading]); // Effect runs when job, user, or auth state changes.

    // Effect for polling while the job is active
    useEffect(() => {
        if (!job || (job.status !== 'PENDING' && job.status !== 'PROGRESS')) {
            return;
        }

        const pollJobStatus = async () => {
            try {
                const fetchedJobData = await getJobById(jobId, token);
                setJob({ ...fetchedJobData, result_payload: fetchedJobData.result });
            } catch (err) {
                // A failed poll shouldn't disrupt the UI with a big error message.
                // We just log it and the next poll will try again.
                console.error(`Polling failed for job ${jobId}:`, err);
            }
        };

        const intervalId = setInterval(pollJobStatus, 5000);
        return () => clearInterval(intervalId);
    }, [job, jobId, token]); // Dependencies are correct

    // Download handler
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

    // Follow-up handler
    const handleFollowUpSubmit = (e) => {
        e.preventDefault();
        if (!followUpPrompt.trim()) return;
        console.log(`Submitting follow-up for job ${jobId}: "${followUpPrompt}"`);
        alert("Follow-up functionality is not yet implemented in the backend.");
        setFollowUpPrompt('');
    };

    // --- FIX: Halt rendering until authentication is resolved to prevent race conditions ---
    if (isAuthLoading) {
        return <div className="loading-container"><h2>Loading...</h2></div>;
    }

    if (isPageLoading) {
        return <div className="loading-container"><h2>Loading Job Details...</h2></div>;
    }

    if (error) {
        return <div className="job-detail-page"><div className="error-text">Error: {error}</div></div>;
    }

    if (!job) {
        return <div className="job-detail-page"><p>Job not found.</p></div>;
    }

    return (
        <div className="job-detail-page">
            <div className="conversation-thread">
                {/* Card 1: User's Request */}
                <div className="task-card user-card">
                    <div className="card-header">
                        <span className="card-author">Your Request</span>
                        <span className="card-timestamp">{formatDate(job.created_at)}</span>
                    </div>
                    <div className="card-body">
                        {/* This now renders correctly because the API provides the 'prompt' field */}
                        <p>"{job.prompt}"</p>
                    </div>
                </div>

                {/* Card 2: Agent's Result */}
                <div className={`task-card agent-card status-${job.status.toLowerCase()}`}>
                    <div className="card-header">
                        <span className="card-author">Codec Agent</span>
                        <span className="card-status">{getStatusText(job.status)}</span>
                    </div>
                    <div className="card-body">
                        {(job.status === 'PENDING' || job.status === 'PROGRESS') && (
                            <div className="working-state">
                                <div className="spinner-large"></div>
                                <h3>Agent is working...</h3>
                                <p>This may take a few minutes. You can close this page and come back later.</p>
                            </div>
                        )}
                        {job.status === 'SUCCESS' && (
                            <div className="success-state">
                                <p>{job.result_payload?.message || 'Your edit is complete!'}</p>
                                {job.result_payload?.output_path && (
                                    <div className="attachments">
                                        <h4>Attachments</h4>
                                        <button onClick={handleDownloadClick} className="download-button">
                                            Download Edit
                                        </button>
                                    </div>
                                )}
                            </div>
                        )}
                        {job.status === 'FAILURE' && (
                            <div className="failure-state">
                                <p>{job.result_payload?.message || 'An unexpected error occurred and the job could not be completed.'}</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Card 3: Follow-up Input */}
                <div className="task-card follow-up-card">
                     <form onSubmit={handleFollowUpSubmit}>
                        <textarea
                            value={followUpPrompt}
                            onChange={(e) => setFollowUpPrompt(e.target.value)}
                            placeholder="Ask a follow-up (e.g., 'Make it shorter' or 'Change the music')..."
                            rows="1"
                            disabled={job.status !== 'SUCCESS'}
                        />
                        <button type="submit" disabled={!followUpPrompt.trim() || job.status !== 'SUCCESS'}>
                            Send
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default JobDetailPage;