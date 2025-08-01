// frontend/src/components/JobList.js

import React from 'react';
import JobItem from './JobItem';
import './JobList.css';

function JobList({ jobs, isLoading, error }) {
    
    const renderContent = () => {
        if (isLoading) {
            return <p className="info-text">Loading your edits...</p>;
        }

        if (error) {
            return <p className="error-text">Error: {error}</p>;
        }

        if (!jobs || jobs.length === 0) {
            return (
                <div className="no-jobs-container">
                    <h3>Welcome to Codec!</h3>
                    <p>You haven't created any edits yet. Use the form above to start your first one.</p>
                </div>
            );
        }

        return (
            <div className="job-list">
                {jobs.map(job => (
                    <JobItem key={job.job_id} initialJob={job} />
                ))}
            </div>
        );
    };

    return (
        <div className="job-list-container">
            <h2>My Edits</h2>
            {renderContent()}
        </div>
    );
}

export default JobList;