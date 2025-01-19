import { ChangeEvent, useRef, useEffect, useState } from 'react';
import { Button, TextField, Typography } from '@mui/material';
import { AccountCircle, SmartToy } from '@mui/icons-material';
import { useAuth } from './auth/AuthProvider';
import { Navigate } from 'react-router-dom';
import { ChatRecord } from './auth/auth-interface';
import CircularProgress from '@mui/material/CircularProgress';
import './App.css';

function App() {
  const [question, setQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState<ChatRecord[]>([])
  const [runningIngestion, setRunningIngestion] = useState(false)
  const messageEndRef = useRef(null)
  const auth = useAuth()
  const baseUrl = `${import.meta.env.VITE_API_URL}/api/chatbot`;

  useEffect(() => {
    if (auth.token === '') {
      return
    }
    const load_chat_history = async () => {
      const res = await fetch(`${baseUrl}/chat-history`, {
        method: "GET",
        headers: { "content-Type": "application/json", Authorization: `Bearer ${auth.token}` },
      });
      const data = await res.json();
      setChatHistory(data.chat_history)
    }
    load_chat_history()
  }, [auth.token, baseUrl])

  useEffect(() => {
    if (messageEndRef?.current && chatHistory?.length) {
      console.log(messageEndRef);

      messageEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [messageEndRef, chatHistory?.length])

  if (auth.token === '') {
    return <Navigate to='/login' />
  }

  const handleChangeQuestion = (event: ChangeEvent<HTMLInputElement>) => {
    setQuestion(event.target.value)
  }

  const handleKeyPress = async (event: { key: string; }) => {
    if (event.key === 'Enter' && question.trim() !== '') {
      const res = await fetch(`${baseUrl}/completion`, {
        method: "POST",
        headers: { "content-Type": "application/json", Authorization: `Bearer ${auth.token}` },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setChatHistory([
        ...chatHistory,
        { role_type: "human", content: question },
        { role_type: "ai", content: data.completion },
      ])
      setQuestion('')
    }
  }

  const handleGenKnowledgeBase = async () => {
    setRunningIngestion(true)
    const res = await fetch(`${baseUrl}/gen-knowledgebase`, {
      method: "POST",
      headers: { Authorization: `Bearer ${auth.token}` },
    });
    if (res.status === 200) {
      setRunningIngestion(false)
      alert("Successfully generate knowledgebase in backend!");
    }
  }

  const handleLogout = () => {
    auth.logout()
  }

  return (
    <div className='main-container'>
      <div className='title-container'>
        <Button
          variant='contained'
          size='small'
          onClick={handleGenKnowledgeBase}
          className='ingestion-button'
        >
          {runningIngestion ? <CircularProgress size='20px' color='inherit' /> : 'Generate Knowledge Base'}
        </Button>
        <h2 className='title'>Finance Chatbot</h2>
        <div className='user-section'>
          <Typography className='user-greeting'>Hi, {auth.user}</Typography>
          <Button
            variant='contained'
            size='small'
            onClick={handleLogout}
          >
            Logout
          </Button>
        </div>
      </div>
      <div className='chat-history'>
        {chatHistory?.map((record, index) => (
          <div key={index} className='chat-container'>
            {record.role_type === "ai" ? <SmartToy className='ai-logo' /> : <AccountCircle className='human-logo' />}
            <Typography className='chat-content'>{record.content}</Typography>
          </div>
        ))}
        <div className='virtual-div' ref={messageEndRef} />
      </div>
      <div className="question-container">
        <AccountCircle className='question-logo' />
        <TextField
          fullWidth
          value={question}
          className='text-field'
          size='small'
          onChange={handleChangeQuestion}
          onKeyDown={handleKeyPress}
        />
      </div>
    </div>
  )
}

export default App