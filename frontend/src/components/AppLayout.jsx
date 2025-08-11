// frontend/src/components/AppLayout.jsx

import React, { useState } from 'react';
import { Outlet, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import Sidebar from './Sidebar.jsx';
import './AppLayout.css';

function AppLayout() {
    const { user, logout } = useAuth();
    const [isSidebarVisible, setIsSidebarVisible] = useState(true);

    const toggleSidebar = () => {
        setIsSidebarVisible(!isSidebarVisible);
    };

    return (
        <div className="app-layout">
            <Sidebar isVisible={isSidebarVisible} />
            
            <div className="main-content-area">
                <header className="app-header">
                    <div className="header-left">
                        {/* --- MODIFIED BUTTON --- */}
                        <button 
                            onClick={toggleSidebar} 
                            className="sidebar-toggle-button"
                            aria-label={isSidebarVisible ? "Collapse sidebar" : "Expand sidebar"}
                        >
                            {/* 
                                Using an SVG for the icon ensures perfect centering and scalability.
                                The rotation is handled via an inline style for simplicity,
                                but the transition is handled in CSS for better performance.
                            */}
                            <svg 
                                xmlns="http://www.w3.org/2000/svg" 
                                width="1em" 
                                height="1em" 
                                viewBox="0 0 24 24" 
                                fill="none" 
                                stroke="currentColor" 
                                strokeWidth="2.5" 
                                strokeLinecap="round" 
                                strokeLinejoin="round"
                                className="toggle-arrow-svg"
                                style={{ transform: isSidebarVisible ? 'rotate(0deg)' : 'rotate(180deg)' }}
                            >
                                <polyline points="15 18 9 12 15 6"></polyline>
                            </svg>
                        </button>
                        {/* --- END MODIFIED BUTTON --- */}

                        <Link to="/" className="logo">Codec</Link>
                    </div>
                    <div className="user-info">
                        <span>{user?.email}</span>
                        <button onClick={logout} className="logout-button">Logout</button>
                    </div>
                </header>

                <main className="page-content">
                    <Outlet />
                </main>
            </div>
        </div>
    );
}

export default AppLayout;