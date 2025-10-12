import React from 'react';

const WelcomeScreen = ({ onStartChat }) => {
  const suggestions = [
    "Jaké jsou nejnovější vyhlášky?",
    "Co je nového v Děčíně?",
    "Najdi informace o stavebních povoleních",
    "Ukáž mi dopravní omezení"
  ];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      padding: '2rem',
      textAlign: 'center'
    }}>
      <div style={{
        fontSize: '2.5rem',
        marginBottom: '1rem',
        color: '#6b7280'
      }}>
        🏛️
      </div>
      
      <h2 style={{
        fontSize: '1.5rem',
        fontWeight: '600',
        color: '#1a1a1a',
        marginBottom: '0.5rem',
        letterSpacing: '-0.01em'
      }}>
        Vítejte v AI asistentu
      </h2>
      
      <p style={{
        color: '#6b7280',
        fontSize: '0.9rem',
        marginBottom: '2rem',
        maxWidth: '500px',
        lineHeight: '1.5',
        fontWeight: '400'
      }}>
        Pomohu vám najít informace z úředních desek českých obcí. 
        Začněte psaním dotazu nebo vyberte jeden z návrhů níže.
      </p>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '0.75rem',
        width: '100%',
        maxWidth: '600px'
      }}>
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onStartChat(suggestion)}
            style={{
              padding: '0.75rem 1rem',
              background: '#f9fafb',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              color: '#1a1a1a',
              fontSize: '0.85rem',
              cursor: 'pointer',
              transition: 'background-color 0.2s ease',
              textAlign: 'left',
              lineHeight: '1.4'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#f3f4f6';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = '#f9fafb';
            }}
          >
            💬 {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default WelcomeScreen;