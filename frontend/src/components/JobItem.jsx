// frontend/src/components/JobItem.jsx

import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import { getJobStatus, getDownloadUrl } from '../services/api';
import './JobItem.css';

const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
};

function JobItem({ initialJob }) {
    const [job, setJob] = useState(initialJob);
    const { token } = useAuth();

    useEffect(() => {
        const isRunning = job.status === 'PENDING' || job.status === 'PROGRESS';

        // Only start polling if the job is in a running state.
        if (!isRunning) {
            return;
        }

        const intervalId = setInterval(async () => {
            try {
                // This API endpoint now returns the full job object from the DB.
                const updatedJobData = await getJobStatus(job.job_id, token);
                
                // Simply update the entire job state with the new data from the DB.
                // This is simpler and ensures all fields (status, result_payload) are in sync.
                // The API result has a 'result' key which maps to our 'result_payload'.
                setJob(prevJob => ({
                    ...prevJob,
                    status: updatedJobData.status,
                    result_payload: updatedJobData.result,
                }));

            } catch (error) {
                console.error(`Failed to get status for job ${job.job_id}`, error);
                // On a polling error, we assume the job failed to prevent spamming a broken endpoint.
                setJob(prevJob => ({ ...prevJob, status: 'FAILURE', result_payload: { message: 'Error fetching status.' } }));
            }
        }, 5000); // Poll every 5 seconds

        // The cleanup function is crucial. It runs when the component unmounts
        // or when the dependencies in the array below change.
        return () => {
            clearInterval(intervalId);
        };
        // The effect re-runs if the job's status changes. This correctly handles
        // the transition from a running state to a final state, at which point
        // the `if (!isRunning)` check will stop the polling.
    }, [job.status, job.job_id, token]);

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
                    {/* Use the payload from the job state, which is now correctly updated */}
                    {job.result_payload?.message || 'Waiting for agent to start...'}
                </p>
            </div>
            <div className="job-item-footer">
                <p className="job-date">Created: {formatDate(job.created_at)}</p>
                {job.status === 'SUCCESS' && job.result_payload?.output_path && (
                    <a
                        href={getDownloadUrl(job.job_id)}
                        className="download-button"
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => {
                            // This fetch-based download is the correct way to handle
                            // authenticated downloads, as it allows sending the Auth header.
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
                                    const filename = job.result_payload.output_path.split('/').pop();
                                    a.download = filename || 'codec_edit.otio';
                                    document.body.appendChild(a);
                                    a.click();
                                    window.URL.revokeObjectURL(url);
                                    a.remove();
                                })
                                .catch(err => {
                                    console.error("Download error:", err);
                                    // Optionally, show an error to the user here.
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