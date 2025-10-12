import { render, screen } from '@testing-library/react';
import App from './App';

test('renders chatbot header', () => {
  render(<App />);
  const headerElement = screen.getByText(/Úřední Deska Chatbot/i);
  expect(headerElement).toBeInTheDocument();
});

test('renders chat interface', () => {
  render(<App />);
  const inputElement = screen.getByPlaceholderText(/Napište vaši otázku/i);
  expect(inputElement).toBeInTheDocument();
});

test('renders sidebar', () => {
  render(<App />);
  const sidebarElement = screen.getByText(/Stav systému/i);
  expect(sidebarElement).toBeInTheDocument();
});