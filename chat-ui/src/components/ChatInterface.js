import React, { useState, useRef, useEffect } from 'react';
import {
    Send, Paperclip, MoreHorizontal, CheckCircle, XCircle,
    Info, Check, Search, Smile, Image, Mic, Clock
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './ChatInterface.css';

const BACKEND_ROUTE = 'api/routes/chat/'

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        {
            id: 'welcome-msg',
            text: "Hi, I'm Agent Pugo Hilion! I'm a social AI agent fine-tuned on Hugo Philion's tweets. How can I assist you today?",
            type: 'bot',
            timestamp: new Date(),
            status: 'delivered'
        }
    ]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);
    const [pendingTransaction, setPendingTransaction] = useState(null);
    const [showInfo, setShowInfo] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        // Focus the input field when the component mounts
        inputRef.current?.focus();
    }, []);

    const handleSendMessage = async (text) => {
        try {
            const response = await fetch(BACKEND_ROUTE, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: text }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // Check if response contains a transaction preview
            if (data.response.includes('Transaction Preview:')) {
                setAwaitingConfirmation(true);
                setPendingTransaction(text);
            }

            // Add a small delay for a more natural conversation feel
            await new Promise(resolve => setTimeout(resolve, 300));
            return data.response;
        } catch (error) {
            console.error('Error:', error);
            return 'Sorry, there was an error processing your request. Please try again.';
        }
    };

    const [searchActive, setSearchActive] = useState(false);
    const [showEmojiPicker, setShowEmojiPicker] = useState(false);
    const [groupMessagesByDate, setGroupMessagesByDate] = useState(true);

    // Generate unique message ID
    const generateMessageId = () => `msg-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

    // Group messages by date for display
    const getGroupedMessages = () => {
        if (!groupMessagesByDate) return { ungrouped: messages };

        return messages.reduce((groups, message) => {
            const date = new Date(message.timestamp);
            const dateStr = date.toDateString();

            if (!groups[dateStr]) {
                groups[dateStr] = [];
            }

            groups[dateStr].push(message);
            return groups;
        }, {});
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!inputText.trim() || isLoading) return;

        const messageText = inputText.trim();
        setInputText('');
        setIsLoading(true);

        const messageId = generateMessageId();

        // Add user message with animation and initial "sending" status
        setMessages(prev => [...prev, {
            id: messageId,
            text: messageText,
            type: 'user',
            isNew: true,
            timestamp: new Date(),
            status: 'sending'
        }]);

        // Update to "sent" status after a short delay
        setTimeout(() => {
            setMessages(prev =>
                prev.map(msg =>
                    msg.id === messageId ? { ...msg, status: 'sent', isNew: false } : msg
                )
            );
        }, 300);

        // Handle transaction confirmation
        if (awaitingConfirmation) {
            if (messageText.toUpperCase() === 'CONFIRM') {
                setAwaitingConfirmation(false);
                const response = await handleSendMessage(pendingTransaction);
                const botMsgId = generateMessageId();

                setMessages(prev => [...prev, {
                    id: botMsgId,
                    text: response,
                    type: 'bot',
                    isNew: true,
                    timestamp: new Date(),
                    status: 'delivered'
                }]);

                // Mark user message as "delivered" and "read" after bot responds
                setMessages(prev =>
                    prev.map(msg =>
                        msg.id === messageId ? { ...msg, status: 'read' } : msg
                    )
                );

                // Remove "isNew" flag after animation completes
                setTimeout(() => {
                    setMessages(prev =>
                        prev.map(msg =>
                            msg.id === botMsgId ? { ...msg, isNew: false } : msg
                        )
                    );
                }, 300);
            } else {
                setAwaitingConfirmation(false);
                setPendingTransaction(null);

                const botMsgId = generateMessageId();
                setMessages(prev => [...prev, {
                    id: botMsgId,
                    text: 'Transaction cancelled. How else can I help you?',
                    type: 'bot',
                    isNew: true,
                    timestamp: new Date(),
                    status: 'delivered'
                }]);

                // Mark user message as "delivered" and "read" after bot responds
                setMessages(prev =>
                    prev.map(msg =>
                        msg.id === messageId ? { ...msg, status: 'read' } : msg
                    )
                );

                setTimeout(() => {
                    setMessages(prev =>
                        prev.map(msg =>
                            msg.id === botMsgId ? { ...msg, isNew: false } : msg
                        )
                    );
                }, 300);
            }
        } else {
            const response = await handleSendMessage(messageText);
            const botMsgId = generateMessageId();

            setMessages(prev => [...prev, {
                id: botMsgId,
                text: response,
                type: 'bot',
                isNew: true,
                timestamp: new Date(),
                status: 'delivered'
            }]);

            // Mark user message as "delivered" and "read" after bot responds
            setMessages(prev =>
                prev.map(msg =>
                    msg.id === messageId ? { ...msg, status: 'read' } : msg
                )
            );

            setTimeout(() => {
                setMessages(prev =>
                    prev.map(msg =>
                        msg.id === botMsgId ? { ...msg, isNew: false } : msg
                    )
                );
            }, 300);
        }

        setIsLoading(false);
        inputRef.current?.focus();
    };

    // Custom components for ReactMarkdown
    const MarkdownComponents = {
        // Override paragraph to remove default margins
        p: ({ children }) => <span className="inline">{children}</span>,
        // Style code blocks
        code: ({ node, inline, className, children, ...props }) => (
            inline ?
                <code className="inline-code">{children}</code> :
                <pre className="code-block">
          <code {...props}>{children}</code>
        </pre>
        ),
        // Style links
        a: ({ node, children, ...props }) => (
            <a {...props} className="chat-link">{children}</a>
        )
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    // Format date for display
    const formatDate = (date) => {
        const now = new Date();
        const messageDate = new Date(date);

        // Check if it's today
        if (messageDate.toDateString() === now.toDateString()) {
            return 'Today';
        }

        // Check if it's yesterday
        const yesterday = new Date(now);
        yesterday.setDate(now.getDate() - 1);
        if (messageDate.toDateString() === yesterday.toDateString()) {
            return 'Yesterday';
        }

        // Otherwise return formatted date
        return messageDate.toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'short',
            day: 'numeric'
        });
    };

    // Format time for display
    const formatTime = (date) => {
        return new Date(date).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    // Render message status indicators (double ticks, etc.)
    const renderMessageStatus = (status) => {
        switch(status) {
            case 'sending':
                return <Clock size={14} className="message-status-icon sending" />;
            case 'sent':
                return <Check size={14} className="message-status-icon sent" />;
            case 'delivered':
                return (
                    <div className="double-check">
                        <Check size={14} className="message-status-icon delivered" />
                        <Check size={14} className="message-status-icon delivered" />
                    </div>
                );
            case 'read':
                return (
                    <div className="double-check">
                        <Check size={14} className="message-status-icon read" />
                        <Check size={14} className="message-status-icon read" />
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="chat-container">
            {/* Chat header */}
            <div className="chat-header">
                <div className="chat-header-info">
                    <div className="chat-avatar">
                        <span>PH</span>
                    </div>
                    <div>
                        <h2>Agent Pugo Hilion</h2>
                        <div className="status-indicator">
                            <span className="status-dot"></span>
                            <span>Online</span>
                        </div>
                    </div>
                </div>
                <div className="chat-header-actions">
                    <button
                        className={`icon-button ${searchActive ? 'active' : ''}`}
                        onClick={() => setSearchActive(!searchActive)}
                        aria-label="Search messages"
                    >
                        <Search size={18} />
                    </button>
                    <button
                        className="icon-button"
                        onClick={() => setShowInfo(!showInfo)}
                        aria-label="Show information"
                    >
                        <Info size={18} />
                    </button>
                    <button className="icon-button" aria-label="More options">
                        <MoreHorizontal size={18} />
                    </button>
                </div>
            </div>

            {/* Search bar */}
            {searchActive && (
                <div className="search-bar">
                    <Search size={16} className="search-icon" />
                    <input
                        type="text"
                        placeholder="Search in conversation..."
                        className="search-input"
                    />
                    <button
                        className="search-close"
                        onClick={() => setSearchActive(false)}
                    >
                        <XCircle size={16} />
                    </button>
                </div>
            )}

            {/* Info panel */}
            {showInfo && (
                <div className="info-panel">
                    <div className="info-content">
                        <h3>About this AI</h3>
                        <p>Agent Pugo Hilion is a social AI agent fine-tuned on Hugo Philion's tweets. It runs in a Trusted Execution Environment (TEE) for secure AI execution.</p>
                        <p>This AI is powered by Gemini 1.5 Flash and supports markdown in its responses.</p>
                    </div>
                    <button className="close-info" onClick={() => setShowInfo(false)}>
                        <XCircle size={16} />
                    </button>
                </div>
            )}

            {/* Messages container */}
            <div className="messages-container">
                {groupMessagesByDate
                    ? Object.entries(getGroupedMessages()).map(([dateStr, msgs]) => (
                        <div key={dateStr} className="message-group">
                            <div className="date-separator">
                                <span>{formatDate(dateStr)}</span>
                            </div>
                            {msgs.map((message) => (
                                <div
                                    key={message.id}
                                    className={`message-wrapper ${message.type === 'user' ? 'user-message-wrapper' : 'bot-message-wrapper'} ${message.isNew ? 'message-new' : ''}`}
                                >
                                    {message.type === 'bot' && (
                                        <div className="avatar bot-avatar">
                                            <span>PH</span>
                                        </div>
                                    )}
                                    <div
                                        className={`message ${message.type === 'user' ? 'user-message' : 'bot-message'}`}
                                    >
                                        <ReactMarkdown
                                            components={MarkdownComponents}
                                            className="message-content"
                                        >
                                            {message.text}
                                        </ReactMarkdown>
                                        <div className="message-footer">
                                            <div className="message-time">
                                                {formatTime(message.timestamp)}
                                            </div>
                                            {message.type === 'user' && (
                                                <div className="message-status">
                                                    {renderMessageStatus(message.status)}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                    {message.type === 'user' && (
                                        <div className="avatar user-avatar">
                                            <span>U</span>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    ))
                    : messages.map((message) => (
                        <div
                            key={message.id}
                            className={`message-wrapper ${message.type === 'user' ? 'user-message-wrapper' : 'bot-message-wrapper'} ${message.isNew ? 'message-new' : ''}`}
                        >
                            {message.type === 'bot' && (
                                <div className="avatar bot-avatar">
                                    <span>PH</span>
                                </div>
                            )}
                            <div
                                className={`message ${message.type === 'user' ? 'user-message' : 'bot-message'}`}
                            >
                                <ReactMarkdown
                                    components={MarkdownComponents}
                                    className="message-content"
                                >
                                    {message.text}
                                </ReactMarkdown>
                                <div className="message-footer">
                                    <div className="message-time">
                                        {formatTime(message.timestamp)}
                                    </div>
                                    {message.type === 'user' && (
                                        <div className="message-status">
                                            {renderMessageStatus(message.status)}
                                        </div>
                                    )}
                                </div>
                            </div>
                            {message.type === 'user' && (
                                <div className="avatar user-avatar">
                                    <span>U</span>
                                </div>
                            )}
                        </div>
                    ))}
                {isLoading && (
                    <div className="message-wrapper bot-message-wrapper">
                        <div className="avatar bot-avatar">
                            <span>PH</span>
                        </div>
                        <div className="message bot-message loading-message">
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Transaction confirmation bar */}
            {awaitingConfirmation && (
                <div className="confirmation-bar">
                    <div className="confirmation-content">
                        <Info size={18} />
                        <span>Transaction requires confirmation. Type CONFIRM to proceed or anything else to cancel.</span>
                    </div>
                    <div className="confirmation-actions">
                        <button className="confirm-button" onClick={() => {
                            setInputText('CONFIRM');
                            handleSubmit({ preventDefault: () => {} });
                        }}>
                            <CheckCircle size={16} />
                            <span>Confirm</span>
                        </button>
                        <button className="cancel-button" onClick={() => {
                            setInputText('CANCEL');
                            handleSubmit({ preventDefault: () => {} });
                        }}>
                            <XCircle size={16} />
                            <span>Cancel</span>
                        </button>
                    </div>
                </div>
            )}

            {/* Input area */}
            <div className="input-area">
                <form onSubmit={handleSubmit} className="input-form">
                    <div className="input-tools">
                        <button type="button" className="tool-button" aria-label="Attach file">
                            <Paperclip size={18} />
                        </button>
                        <button type="button" className="tool-button" aria-label="Send image">
                            <Image size={18} />
                        </button>
                        <button
                            type="button"
                            className={`tool-button ${showEmojiPicker ? 'active' : ''}`}
                            onClick={() => setShowEmojiPicker(!showEmojiPicker)}
                            aria-label="Insert emoji"
                        >
                            <Smile size={18} />
                        </button>
                        <button type="button" className="tool-button" aria-label="Voice message">
                            <Mic size={18} />
                        </button>
                    </div>
                    <textarea
                        ref={inputRef}
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={awaitingConfirmation
                            ? "Type CONFIRM to proceed or anything else to cancel"
                            : "Type your message... (Markdown supported)"}
                        className="input-field"
                        rows="1"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !inputText.trim()}
                        className={`send-button ${isLoading || !inputText.trim() ? 'disabled' : ''}`}
                        aria-label="Send message"
                    >
                        <Send size={18} />
                    </button>
                </form>

                {/* Emoji picker (simplified version) */}
                {showEmojiPicker && (
                    <div className="emoji-picker">
                        <div className="emoji-container">
                            {['👍', '❤️', '😊', '🎉', '👋', '😂', '🙏', '🔥', '👏', '🤔'].map(emoji => (
                                <button
                                    key={emoji}
                                    className="emoji-btn"
                                    onClick={() => {
                                        setInputText(prev => prev + emoji);
                                        setShowEmojiPicker(false);
                                        inputRef.current?.focus();
                                    }}
                                >
                                    {emoji}
                                </button>
                            ))}
                        </div>
                        <button
                            className="emoji-close"
                            onClick={() => setShowEmojiPicker(false)}
                        >
                            <XCircle size={16} />
                        </button>
                    </div>
                )}

                <div className="input-footer">
                    <span className="markdown-hint">Markdown supported. Press Enter to send, Shift+Enter for new line.</span>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;