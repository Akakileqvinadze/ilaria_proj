import React from 'react';
import ChatWidget from './components/ChatWidget';
import './index.css';

const App = () => {
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

  return (
    <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-4">
      <div className="w-full max-w-5xl grid md:grid-cols-[2fr,1.5fr] gap-8 items-start">
        <div className="space-y-6">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            ილარია HR – ინსტრუქციული ჩატბოტი
          </h1>
          <p className="text-base md:text-lg text-white/70 max-w-xl">
            დაუსვი შეკითხვა ილარია HR სისტემის შესახებ და მიიღე სწრაფი,
            გასაგები და ქართულ ენაზე მომზადებული პასუხები.
          </p>
          <div className="space-y-2 text-sm text-white/60">
            <p>🔌 API URL: <span className="font-mono text-xs">{apiUrl}</span></p>
            <ul className="list-disc list-inside space-y-1">
              <li>Frontend → Node.js Backend: <code>/api/chat</code></li>
              <li>Backend → AI Service (FastAPI): <code>/api/chat</code></li>
            </ul>
          </div>
        </div>

        <div className="relative">
          <ChatWidget apiUrl={apiUrl} />
        </div>
      </div>
    </div>
  );
};

export default App;
