import React from 'react';

const Sidebar = ({ systemStatus, documents, onRefresh }) => {
  const formatDate = (dateString) => {
    if (!dateString) return 'Nikdy';
    return new Date(dateString).toLocaleString('cs-CZ');
  };

  const getConnectionStatus = (connected) => {
    return connected ? '✓ Připojeno' : '✗ Nepřipojeno';
  };

  const getConnectionColor = (connected) => {
    return connected ? '#51cf66' : '#ff6b6b';
  };

  const getMunicipalities = (docs) => {
    const municipalityCount = {};
    
    docs.forEach(doc => {
      const municipality = doc.municipality || 'Neznámá obec';
      municipalityCount[municipality] = (municipalityCount[municipality] || 0) + 1;
    });

    return Object.entries(municipalityCount)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count); // Seřadit podle počtu dokumentů
  };

  return (
    <div className="sidebar">
      <h3>Stav systému</h3>
      
      <div className="status-item">
        <span className="status-label">Dokumenty:</span>
        <span className="status-value">{systemStatus.documentsCount}</span>
      </div>
      
      <div className="status-item">
        <span className="status-label">Poslední aktualizace:</span>
        <span className="status-value">{formatDate(systemStatus.lastUpdate)}</span>
      </div>
      
      <div className="status-item">
        <span className="status-label">Ollama:</span>
        <span 
          className="status-value" 
          style={{ color: getConnectionColor(systemStatus.ollamaConnected) }}
        >
          {getConnectionStatus(systemStatus.ollamaConnected)}
        </span>
      </div>

      <button 
        onClick={onRefresh}
        style={{
          width: '100%',
          padding: '0.75rem 1rem',
          marginTop: '1rem',
          background: '#2563eb',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          fontSize: '0.85rem',
          fontWeight: '500',
          transition: 'background-color 0.2s ease'
        }}
        onMouseEnter={(e) => {
          e.target.style.background = '#1d4ed8';
        }}
        onMouseLeave={(e) => {
          e.target.style.background = '#2563eb';
        }}
      >
        Obnovit stav
      </button>

      {documents.length > 0 && (
        <div className="documents-list">
          <h3>Pokryté obce ({getMunicipalities(documents).length})</h3>
          {getMunicipalities(documents).map((municipality, index) => (
            <div key={index} className="document-item">
              <div className="document-title">
                {municipality.name}
              </div>
              <div className="document-municipality">
                {municipality.count} {municipality.count === 1 ? 'dokument' : 
                 municipality.count <= 4 ? 'dokumenty' : 'dokumentů'}
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ 
        marginTop: '1.5rem', 
        padding: '1rem', 
        background: '#f9fafb',
        borderRadius: '8px',
        border: '1px solid #e5e7eb'
      }}>
        <h4 style={{ 
          margin: '0 0 0.75rem 0', 
          fontSize: '0.9rem', 
          color: '#1a1a1a',
          fontWeight: '600',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          💡 Nápověda
        </h4>
        <ul style={{ 
          fontSize: '0.8rem', 
          lineHeight: '1.5', 
          color: '#6b7280',
          listStyle: 'none',
          paddingLeft: '0'
        }}>
          <li style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <span style={{ color: '#2563eb' }}>•</span>
            Ptejte se na informace z úředních desek
          </li>
          <li style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <span style={{ color: '#2563eb' }}>•</span>
            Zkuste: "Jaké jsou nejnovější vyhlášky?"
          </li>
          <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <span style={{ color: '#2563eb' }}>•</span>
            Nebo: "Co je nového v Děčíně?"
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Sidebar;