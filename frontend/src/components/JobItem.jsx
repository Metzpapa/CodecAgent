// frontend/src/components/JobItem.jsx

import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import { getJobStatus, getDownloadUrl } from '../services/api';
import './JobItem.css';

const formatDate = (dateString) => {
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
                const updatedStatus = await getJobStatus(job.job_id, token);
                
                const newStatus = updatedStatus.status;
                const newResultPayload = updatedStatus.result;

                // When the status changes, update the entire job object in our state.
                // This ensures that when the job finishes, we get the final
                // status AND the result_payload in the same update.
                setJob(prevJob => ({
                    ...prevJob,
                    status: newStatus,
                    // Only update the payload if it's a final state and the payload exists
                    ...( (newStatus === 'SUCCESS' || newStatus === 'FAILURE') && newResultPayload 
                         ? { result_payload: newResultPayload } 
                         : {}
                       )
                }));

            } catch (error) {
                console.error(`Failed to get status for job ${job.job_id}`, error);
                // On a polling error, we assume the job failed to prevent spamming a broken endpoint.
                // We also stop the interval here.
                setJob(prevJob => ({ ...prevJob, status: 'FAILURE', result_payload: { message: 'Error fetching status.' } }));
            }
        }, 5000); // Poll every 5 seconds

        // The cleanup function is crucial. It runs when the component unmounts
        // OR when the dependencies in the array below change.
        return () => {
            clearInterval(intervalId);
        };
        // The effect should re-run if the job's status changes.
        // This correctly handles the transition from 'PROGRESS' to a final state,
        // at which point the `if (!isRunning)` check will cause the effect to exit
        // and no new interval will be created.
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
                            e.preventDefault();
                            fetch(getDownloadUrl(job.job_id), { headers: { 'Authorization': `Bearer ${token}` } })
                                .then(res => res.blob())
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