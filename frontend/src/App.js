import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import './index.css';

function App() {
  const [systemStatus, setSystemStatus] = useState({
    documentsCount: 0,
    lastUpdate: null,
    ollamaConnected: false
  });

  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    // Načteme počáteční stav systému
    fetchSystemStatus();
    fetchDocuments();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/status');
      if (response.ok) {
        const status = await response.json();
        setSystemStatus(status);
      }
    } catch (error) {
      console.error('Chyba při načítání stavu systému:', error);
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await fetch('/api/documents');
      if (response.ok) {
        const docs = await response.json();
        setDocuments(docs);
      }
    } catch (error) {
      console.error('Chyba při načítání dokumentů:', error);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Úřední Deska Chatbot</h1>
        <p>AI asistent pro informace z úředních desek vybraných českých obcí</p>
      </header>
      
      <div className="main-container">
        <ChatInterface />
        <Sidebar 
          systemStatus={systemStatus}
          documents={documents}
          onRefresh={() => {
            fetchSystemStatus();
            fetchDocuments();
          }}
        />
      </div>
    </div>
  );
}

export default App;