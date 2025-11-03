import React from 'react';
import ReactDOM from 'react-dom/client';
import { ChatInterface } from './components/ChatInterface';
import './index.scss';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ChatInterface />
  </React.StrictMode>,
);