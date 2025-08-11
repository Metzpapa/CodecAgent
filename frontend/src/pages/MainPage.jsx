// frontend/src/pages/MainPage.jsx

import React from 'react';
import CreateJobForm from '../components/CreateJobForm.jsx';
import './MainPage.css'; // This CSS file will also be updated

function MainPage() {
    // This component no longer needs to fetch jobs or handle user info.
    // Its only purpose is to provide a welcoming page for creating a new job.
    // The logic to navigate after job creation will be moved into CreateJobForm itself.

    return (
        <div className="main-page-container">
            <main className="main-content">
                {/* A dedicated, centered section for creating a job */}
                <section className="creation-section">
                    <h2 className="creation-title">Create an edit with Codec...</h2>
                    {/* 
                      This form will be updated next to handle its own navigation
                      after a job is created. The onJobCreated prop is no longer needed here.
                    */}
                    <CreateJobForm />
                </section>
            </main>
        </div>
    );
}

export default MainPage;