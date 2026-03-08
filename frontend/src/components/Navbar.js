import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import './Navbar.css';

const BACKEND = 'http://localhost:8000';

export default function Navbar() {
    const [online, setOnline] = useState(null);

    useEffect(() => {
        let mounted = true;
        const check = () =>
            fetch(`${BACKEND}/health`, { signal: AbortSignal.timeout(3000) })
                .then(r => r.ok ? 'online' : 'error')
                .catch(() => 'offline')
                .then(s => mounted && setOnline(s));
        check();
        const id = setInterval(check, 15000);
        return () => { mounted = false; clearInterval(id); };
    }, []);

    const links = [
        { to: '/', label: 'Home' },
        { to: '/architecture', label: 'Architecture' },
        { to: '/metrics', label: 'Performance' },
        { to: '/about', label: 'About' },
    ];

    return (
        <nav className="navbar">
            <div className="navbar-inner">
                <NavLink to="/" className="navbar-brand">
                    <span className="brand-icon">🧠</span>
                    <span className="brand-text">Med<span className="brand-accent">AI</span></span>
                </NavLink>

                <ul className="navbar-links">
                    {links.map(({ to, label }) => (
                        <li key={to}>
                            <NavLink
                                to={to}
                                className={({ isActive }) =>
                                    'nav-link' + (isActive ? ' active' : '')}
                                end={to === '/'}
                            >
                                {label}
                            </NavLink>
                        </li>
                    ))}
                </ul>

                <div className={`backend-status status-${online}`}>
                    <span className="status-dot" />
                    <span className="status-label">
                        {online === 'online' ? 'API Online' : online === null ? 'Checking…' : 'API Offline'}
                    </span>
                </div>
            </div>
        </nav>
    );
}
