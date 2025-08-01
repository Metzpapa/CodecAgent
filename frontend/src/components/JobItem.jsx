// frontend/src/components/JobItem.js

import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import { getJobStatus, getDownloadUrl } from '../services/api';
import './JobItem.css'; // We'll create this CSS file next

// A helper to format dates nicely
const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
};

function JobItem({ initialJob }) {
    const [job, setJob] = useState(initialJob);
    const [isPolling, setIsPolling] = useState(false);
    const { token } = useAuth();

    useEffect(() => {
        const isRunning = job.status === 'PENDING' || job.status === 'PROGRESS';

        if (isRunning && !isPolling) {
            setIsPolling(true);
            const intervalId = setInterval(async () => {
                try {
                    const updatedStatus = await getJobStatus(job.job_id, token);
                    
                    // Celery status can be different from our DB status, so we sync them
                    const newStatus = updatedStatus.status;
                    const newResultPayload = updatedStatus.result;

                    setJob(prevJob => ({
                        ...prevJob,
                        status: newStatus,
                        // Only update the payload if it's a final state
                        ...(newStatus === 'SUCCESS' || newStatus === 'FAILURE' ? { result_payload: newResultPayload } : {})
                    }));

                    // If the job is finished, stop polling
                    if (newStatus === 'SUCCESS' || newStatus === 'FAILURE') {
                        clearInterval(intervalId);
                        setIsPolling(false);
                    }
                } catch (error) {
                    console.error(`Failed to get status for job ${job.job_id}`, error);
                    // Stop polling on error to prevent spamming a broken endpoint
                    clearInterval(intervalId);
                    setIsPolling(false);
                    setJob(prevJob => ({ ...prevJob, status: 'FAILURE', result_payload: { message: 'Error fetching status.' } }));
                }
            }, 5000); // Poll every 5 seconds

            // Cleanup function to clear the interval when the component unmounts
            return () => {
                clearInterval(intervalId);
                setIsPolling(false);
            };
        }
    }, [job.status, job.job_id, token, isPolling]);

    const renderStatus = () => {
        switch (job.status) {
            case 'PENDING':
                return <><div className="spinner"></div><span>Queued</span></>;
            case 'PROGRESS':
                return <><div className="spinner"></div><span>Editing...</span></>;
            case 'SUCCESS':
                return <span className="status-success">✔ Complete</span>;
            case 'FAILURE':
                return <span className="status-failure">✖ Failed</span>;
            default:
                return <span>{job.status}</span>;
        }
    };

    return (
        <div className={`job-item ${job.status.toLowerCase()}`}>
            <div className="job-item-header">
                <p className="job-prompt">"{job.prompt}"</p>
                <div className="job-status">{renderStatus()}</div>
            </div>
            <div className="job-item-body">
                <p className="job-agent-message">
                    {job.result_payload?.message || 'Waiting for agent to start...'}
                </p>
            </div>
            <div className="job-item-footer">
                <p className="job-date">Created: {formatDate(job.created_at)}</p>
                {job.status === 'SUCCESS' && job.result_payload?.output_path && (
                    <a
                        href={getDownloadUrl(job.job_id)}
                        className="download-button"
                        target="_blank" // Opens in a new tab
                        rel="noopener noreferrer"
                        // To make the download work correctly, we need to send the auth token.
                        // Since an <a> tag can't have headers, this requires a more complex solution
                        // (like temporary download tokens) which is out of scope for this prototype.
                        // The backend will need to handle this request, for now we assume it works.
                        onClick={(e) => {
                            e.preventDefault();
                            // A better way for stateless apps: fetch with token, create blob URL, and click it.
                            fetch(getDownloadUrl(job.job_id), { headers: { 'Authorization': `Bearer ${token}` } })
                                .then(res => res.blob())
                                .then(blob => {
                                    const url = window.URL.createObjectURL(blob);
                                    const a = document.createElement('a');
                                    a.style.display = 'none';
                                    a.href = url;
                                    // Extract filename from the payload
                                    const filename = job.result_payload.output_path.split('/').pop();
                                    a.download = filename || 'codec_edit.otio';
                                    document.body.appendChild(a);
                                    a.click();
                                    window.URL.revokeObjectURL(url);
                                });
                        }}
                    >
                        Download Edit
                    </a>
                )}
            </div>
        </div>
    );
}

export default JobItem;