import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import WelcomeScreen from './WelcomeScreen';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e, messageText = null) => {
    if (e) e.preventDefault();
    
    const message = messageText || inputValue.trim();
    if (!message || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: data.response,
          sources: data.sources || [],
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error('Chyba serveru');
      }
    } catch (error) {
      console.error('Chyba při odesílání zprávy:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Omlouvám se, došlo k chybě při zpracování vaší zprávy. Zkuste to prosím znovu.',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleWelcomeMessage = (message) => {
    handleSubmit(null, message);
  };

  return (
    <div className="chat-container">
      <div className="messages-container">
        {messages.length === 0 && !isLoading ? (
          <WelcomeScreen onStartChat={handleWelcomeMessage} />
        ) : (
          <>
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.type}`}>
            <div className="message-avatar">
              {message.type === 'user' ? 'Vy' : 'AI'}
            </div>
            <div className="message-content">
              {message.type === 'bot' ? (
                <div className="markdown-content">
                  <ReactMarkdown>
                    {message.content}
                  </ReactMarkdown>
                </div>
              ) : (
                message.content
              )}
              {message.sources && message.sources.length > 0 && (
                <div style={{ 
                  marginTop: '0.75rem', 
                  fontSize: '0.8rem', 
                  opacity: 0.7,
                  padding: '0.5rem 0.75rem',
                  background: '#f0f9ff',
                  borderRadius: '6px',
                  border: '1px solid #e0f2fe'
                }}>
                  <span style={{ color: '#2563eb', fontWeight: '600' }}>Zdroje:</span> {message.sources.join(', ')}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-avatar">🤖</div>
            <div className="typing-indicator">
              <span style={{ color: 'rgba(59, 59, 59, 0.7)', fontSize: '0.9rem' }}>
                Zpracovávám vaši otázku
              </span>
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-form">
          <textarea
            className="message-input"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Napište vaši otázku o úředních deskách..."
            disabled={isLoading}
            rows="1"
            style={{ resize: 'none', minHeight: '44px' }}
          />
          <button
            type="submit"
            className="send-button"
            disabled={!inputValue.trim() || isLoading}
          >
            {isLoading ? '' : ''} {isLoading ? 'Odesílám...' : 'Odeslat'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;