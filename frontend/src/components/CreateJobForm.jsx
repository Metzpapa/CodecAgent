// frontend/src/components/CreateJobForm.jsx

import React, { useState, useCallback, useRef } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import { createJob } from '../services/api';
import './CreateJobForm.css'; // This CSS will be updated next

function CreateJobForm({ onJobCreated }) {
    const [prompt, setPrompt] = useState('');
    const [files, setFiles] = useState([]);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState(null);
    const [isDragging, setIsDragging] = useState(false);

    const { token } = useAuth();
    const fileInputRef = useRef(null);

    // --- Drag and Drop Handlers (No changes needed here) ---
    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback(async (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const droppedItems = [...e.dataTransfer.items];
        const droppedFiles = await Promise.all(
            droppedItems.map(item => getFilesFromEntry(item.webkitGetAsEntry()))
        );
        setFiles(prevFiles => [...prevFiles, ...droppedFiles.flat()]);
    }, []);

    // --- File Input Handler (No changes needed here) ---
    const handleFileSelect = (e) => {
        const selectedFiles = Array.from(e.target.files);
        setFiles(prevFiles => [...prevFiles, ...selectedFiles]);
    };

    // --- Form Submission (No changes needed here) ---
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!prompt.trim() || files.length === 0) {
            setError("Please provide a prompt and at least one file.");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            await createJob(prompt, files, token);
            setPrompt('');
            setFiles([]);
            onJobCreated();
        } catch (err) {
            console.error("Failed to create job:", err);
            setError(err.message || "An unknown error occurred.");
        } finally {
            setIsSubmitting(false);
        }
    };

    // --- NEW JSX STRUCTURE ---
    // The root element is now the <form> itself.
    // Labels and the outer container div have been removed.
    return (
        <form onSubmit={handleSubmit} className="create-job-form">
            <div className="form-group">
                <textarea
                    id="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe what you want to edit... e.g., 'Create a 1-minute highlight reel from my stream VOD, focusing on the funny moments. Add some background music.'"
                    rows="4"
                    required
                />
            </div>

            <div className="form-group">
                <div
                    className={`drop-zone ${isDragging ? 'dragging' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current.click()}
                >
                    <p>Drag & drop files or folders here, or click to select.</p>
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileSelect}
                        multiple
                        webkitdirectory="true"
                        style={{ display: 'none' }}
                    />
                </div>
            </div>

            {files.length > 0 && (
                <div className="file-list-container">
                    <h4>Selected Files:</h4>
                    <ul className="file-list">
                        {files.map((file, index) => (
                            <li key={`${file.name}-${index}`}>{file.name}</li>
                        ))}
                    </ul>
                    <button type="button" onClick={() => setFiles([])} className="clear-button">
                        Clear All
                    </button>
                </div>
            )}

            <div className="submit-section">
                <button type="submit" className="submit-button" disabled={isSubmitting || files.length === 0}>
                    {isSubmitting ? 'Starting Job...' : 'Start Editing Job'}
                </button>
                {error && <p className="error-message">{error}</p>}
            </div>
        </form>
    );
}

// Helper function to recursively get files from dropped entries (including folders)
async function getFilesFromEntry(entry) {
    if (!entry) return [];
    
    if (entry.isFile) {
        return new Promise((resolve) => {
            entry.file(file => resolve([file]));
        });
    }

    if (entry.isDirectory) {
        return new Promise((resolve) => {
            const dirReader = entry.createReader();
            dirReader.readEntries(async (entries) => {
                const files = await Promise.all(entries.map(getFilesFromEntry));
                resolve(files.flat());
            });
        });
    }
    return [];
}

export default CreateJobForm;