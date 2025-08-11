// frontend/src/components/AppLayout.jsx

import React, { useState } from 'react';
import { Outlet, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import Sidebar from './Sidebar.jsx'; // We will create this component next
import './AppLayout.css';

function AppLayout() {
    const { user, logout } = useAuth();
    const [isSidebarVisible, setIsSidebarVisible] = useState(true);

    const toggleSidebar = () => {
        setIsSidebarVisible(!isSidebarVisible);
    };

    return (
        <div className="app-layout">
            {/* The Sidebar will be aware of its own visibility to animate in/out */}
            <Sidebar isVisible={isSidebarVisible} />
            
            <div className="main-content-area">
                <header className="app-header">
                    <div className="header-left">
                        {/* This button controls the sidebar's visibility */}
                        <button onClick={toggleSidebar} className="sidebar-toggle-button">
                            {isSidebarVisible ? '‹' : '›'}
                        </button>
                        <Link to="/" className="logo">Codec</Link>
                    </div>
                    <div className="user-info">
                        <span>{user?.email}</span>
                        <button onClick={logout} className="logout-button">Logout</button>
                    </div>
                </header>

                <main className="page-content">
                    {/* Child routes like MainPage and JobDetailPage will be rendered here */}
                    <Outlet />
                </main>
            </div>
        </div>
    );
}

export default AppLayout;