import React, { useState, useRef, useEffect } from 'react';
import logo from "../../Images/logo.png"
import { URL } from './constant';

const MedicalChatbot = () => {
  const [chatSessions, setChatSessions] = useState([
    {
      id: 1,
      title: "Current Chat",
      messages: [
        {
          id: 1,
          text: "Hello! I'm your medical assistant. How can I help you today?",
          sender: 'bot',
          timestamp: new Date()
        }
      ],
      lastActivity: new Date()
    }
  ]);
  
  const [activeChatId, setActiveChatId] = useState(1);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);

  // Handle responsive sidebar
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setSidebarOpen(true);
      } else {
        setSidebarOpen(false);
      }
    };

    handleResize(); // Set initial state
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
// Scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatSessions, activeChatId]);

  const getActiveChat = () => {
    return chatSessions.find(chat => chat.id === activeChatId);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const activeChat = getActiveChat();
    const newMessage = {
      id: activeChat.messages.length + 1,
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    // Update chat sessions with user message
    const updatedSessions = chatSessions.map(chat => 
      chat.id === activeChatId 
        ? {
            ...chat,
            messages: [...chat.messages, newMessage],
            lastActivity: new Date(),
            title: chat.title === "Current Chat" ? inputMessage.slice(0, 30) + "..." : chat.title
          }
        : chat
    );

    setChatSessions(updatedSessions);
    setInputMessage('');
    setIsTyping(true);

    try {
      // Replace with your actual API endpoint
      const response = await fetch(URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputMessage
        })
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();
      
      // Use the answer from your API response
      const botResponse = {
        id: activeChat.messages.length + 2,
        text: data.answer, // This matches your API response structure
        sender: 'bot',
        timestamp: new Date()
      };

      const sessionsWithBotResponse = updatedSessions.map(chat => 
        chat.id === activeChatId 
          ? {
              ...chat,
              messages: [...chat.messages, botResponse],
              lastActivity: new Date()
            }
          : chat
      );

      setChatSessions(sessionsWithBotResponse);

    } catch (error) {
      console.error('Error calling API:', error);
      
      // Fallback message if API fails
      const errorResponse = {
        id: activeChat.messages.length + 2,
        text: "I'm sorry, I'm having trouble connecting right now. Please try again later.",
        sender: 'bot',
        timestamp: new Date()
      };

      const sessionsWithErrorResponse = updatedSessions.map(chat => 
        chat.id === activeChatId 
          ? {
              ...chat,
              messages: [...chat.messages, errorResponse],
              lastActivity: new Date()
            }
          : chat
      );

      setChatSessions(sessionsWithErrorResponse);
    } finally {
      setIsTyping(false);
    }
  };

  const createNewChat = () => {
    const newChatId = Math.max(...chatSessions.map(chat => chat.id)) + 1;
    const newChat = {
      id: newChatId,
      title: "New Chat",
      messages: [
        {
          id: 1,
          text: "Hello! I'm your medical assistant. How can I help you today?",
          sender: 'bot',
          timestamp: new Date()
        }
      ],
      lastActivity: new Date()
    };
    
    setChatSessions(prev => [newChat, ...prev]);
    setActiveChatId(newChatId);
    
    // Close sidebar on mobile after creating new chat
    if (window.innerWidth < 1024) {
      setSidebarOpen(false);
    }
  };

  const deleteChat = (chatId, e) => {
    e.stopPropagation();
    
    if (chatSessions.length === 1) {
      alert("Cannot delete the only chat session!");
      return;
    }

    const updatedSessions = chatSessions.filter(chat => chat.id !== chatId);
    setChatSessions(updatedSessions);
    
    // If deleting active chat, switch to another one
    if (chatId === activeChatId) {
      setActiveChatId(updatedSessions[0].id);
    }
  };

  const clearCurrentChat = () => {
    const updatedSessions = chatSessions.map(chat => 
      chat.id === activeChatId 
        ? {
            ...chat,
            messages: [
              {
                id: 1,
                text: "Hello! I'm your medical assistant. How can I help you today?",
                sender: 'bot',
                timestamp: new Date()
              }
            ],
            title: "Current Chat"
          }
        : chat
    );
    
    setChatSessions(updatedSessions);
  };

  const handleChatSelect = (chatId) => {
    setActiveChatId(chatId);
    // Close sidebar on mobile after selecting chat
    if (window.innerWidth < 1024) {
      setSidebarOpen(false);
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatDate = (date) => {
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      });
    }
  };

  const activeChat = getActiveChat();

  return (
    <div className="flex h-screen bg-white">
      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed lg:static inset-y-0 left-0 z-30
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} 
        lg:translate-x-0 transition-transform duration-300 ease-in-out
        w-80 bg-gray-50 border-r border-gray-200 
        flex flex-col lg:flex
      `}>
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={createNewChat}
            className="w-full bg-pink-500 text-white py-3 rounded-lg font-semibold hover:bg-pink-600 transition-colors flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2">
          <h3 className="text-sm font-semibold text-gray-500 px-2 py-2">Chat History</h3>
          {chatSessions.map((chat) => (
            <div
              key={chat.id}
              onClick={() => handleChatSelect(chat.id)}
              className={`p-3 rounded-lg mb-2 cursor-pointer transition-colors ${
                chat.id === activeChatId 
                  ? 'bg-pink-500 text-white' 
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1 min-w-0">
                  <p className={`font-medium truncate ${
                    chat.id === activeChatId ? 'text-white' : 'text-gray-900'
                  }`}>
                    {chat.title}
                  </p>
                  <p className={`text-xs mt-1 ${
                    chat.id === activeChatId ? 'text-pink-100' : 'text-gray-500'
                  }`}>
                    {formatDate(chat.lastActivity)}
                  </p>
                  <p className={`text-xs mt-1 ${
                    chat.id === activeChatId ? 'text-pink-200' : 'text-gray-400'
                  }`}>
                    {chat.messages.length} messages
                  </p>
                </div>
                <button
                  onClick={(e) => deleteChat(chat.id, e)}
                  className={`ml-2 p-1 rounded ${
                    chat.id === activeChatId 
                      ? 'text-white hover:bg-pink-600' 
                      : 'text-gray-400 hover:bg-gray-200 hover:text-gray-600'
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="bg-pink-500 text-white p-4 flex justify-between items-center">
          <div className="flex items-center space-x-2 sm:space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-pink-600 rounded-lg transition-colors lg:hidden"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="flex items-center space-x-2 sm:space-x-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-white rounded-full flex items-center justify-center flex-shrink-0">
                <img src={logo} className='rounded-full'/>
              </div>
              <div className="min-w-0">
                <h1 className="text-lg sm:text-xl font-bold truncate">OncoBot</h1>
                <p className="text-pink-100 text-xs sm:text-sm truncate">{activeChat?.title}</p>
              </div>
            </div>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={clearCurrentChat}
              className="bg-white text-pink-500 px-3 py-2 sm:px-4 sm:py-2 rounded-lg font-semibold hover:bg-pink-50 transition-colors text-sm sm:text-base whitespace-nowrap"
            >
              Clear Chat
            </button>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-3 sm:p-4 bg-gray-50">
          {activeChat?.messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === 'user' ? 'justify-end' : 'justify-start'
              } mb-3 sm:mb-4`}
            >
              <div
                className={`max-w-[85%] xs:max-w-xs sm:max-w-md px-3 py-2 sm:px-4 sm:py-3 rounded-2xl ${
                  message.sender === 'user'
                    ? 'bg-pink-500 text-white rounded-br-none'
                    : 'bg-white text-gray-800 rounded-bl-none shadow-sm'
                }`}
              >
                <p className="text-sm sm:text-base">{message.text}</p>
                <p
                  className={`text-xs mt-1 ${
                    message.sender === 'user' ? 'text-pink-200' : 'text-gray-500'
                  }`}
                >
                  {formatTime(message.timestamp)}
                </p>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex justify-start mb-4">
              <div className="bg-white text-gray-800 px-4 py-3 rounded-2xl rounded-bl-none shadow-sm">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Disclaimer */}
        <div className="bg-yellow-50 border-t border-yellow-200 p-2 sm:p-3">
          <p className="text-xs text-yellow-800 text-center px-2">
            ⚠️ This is for informational purposes only. Consult a healthcare professional for medical advice.
          </p>
        </div>

        {/* Input Area */}
        <form onSubmit={handleSendMessage} className="bg-white border-t p-3 sm:p-4">
          <div className="flex space-x-2">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Describe your symptoms or ask a question..."
              className="flex-1 border border-gray-300 rounded-full px-3 sm:px-4 py-2 sm:py-3 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent text-sm sm:text-base"
            />
            <button
              type="submit"
              disabled={!inputMessage.trim()}
              className="bg-pink-500 text-white rounded-full w-10 h-10 sm:w-12 sm:h-12 flex items-center justify-center hover:bg-pink-600 disabled:bg-pink-300 transition-colors flex-shrink-0"
            >
              <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default MedicalChatbot;