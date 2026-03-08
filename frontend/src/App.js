import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import ArchitecturePage from './pages/ArchitecturePage';
import MetricsPage from './pages/MetricsPage';
import AboutPage from './pages/AboutPage';

export default function App() {
    return (
        <BrowserRouter>
            <Navbar />
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/architecture" element={<ArchitecturePage />} />
                <Route path="/metrics" element={<MetricsPage />} />
                <Route path="/about" element={<AboutPage />} />
            </Routes>
        </BrowserRouter>
    );
}
