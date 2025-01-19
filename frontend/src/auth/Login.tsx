import { Button, TextField, Typography } from '@mui/material';
import { ChangeEvent, useState } from 'react';
import { useAuth } from './AuthProvider';
import './Login.css'

const Login = () => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const auth = useAuth();

    const handleChangeUsername = (event: ChangeEvent<HTMLInputElement>) => {
        setUsername(event.target.value)
    }

    const handleChangePassword = (event: ChangeEvent<HTMLInputElement>) => {
        setPassword(event.target.value)
    }

    const handleSubmit = async () => {
        auth.login(username, password)
    }

    return (
        <div className='container'>
            <Typography variant='h4' className='login-title'>Sign In</Typography>
            <div className='login-form'>
                <TextField
                    value={username}
                    placeholder='Username'
                    onChange={handleChangeUsername}
                    className='text-field'
                />
                <TextField
                    value={password}
                    type="password"
                    placeholder='Password'
                    onChange={handleChangePassword}
                    className='text-field'
                />
                <div className='submit-button'>
                    <Button
                        variant='contained'
                        onClick={handleSubmit}
                    >
                        Submit
                    </Button>
                </div>
            </div>
        </div>
    )
}

export default Login;
