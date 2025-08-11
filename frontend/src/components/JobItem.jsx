// frontend/src/components/JobItem.jsx

import React from 'react';
// NEW: Import Link for navigation
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import { getDownloadUrl } from '../services/api';
import './JobItem.css';

const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
};

// This component is now much simpler. It receives the job data and renders it
// as a link. It no longer polls for updates; the JobDetailPage will handle that.
function JobItem({ initialJob: job }) {
    const { token } = useAuth();

    // The polling useEffect has been removed to improve dashboard performance.
    // This component is now a simple presentational link.

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

    const handleDownloadClick = (e) => {
        // CRUCIAL: Stop the click from bubbling up to the parent <Link> component.
        // This prevents navigating to the detail page when the user only wants to download.
        e.stopPropagation();
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
                // Optionally, show an error to the user here.
            });
    };

    return (
        // The entire item is now a link to the job's detail page.
        // The className is moved here to ensure status-based styling still applies.
        <Link to={`/jobs/${job.job_id}`} className={`job-item ${job.status.toLowerCase()}`}>
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
                        onClick={handleDownloadClick}
                    >
                        Download Edit
                    </a>
                )}
            </div>
        </Link>
    );
}

export default JobItem;