import { createContext, ReactNode, useContext, useState } from "react";
import { Context } from "./auth-interface"
import { getUser, getToken, saveUser, saveToken, cleanToken, cleanUser } from "../utils/token-services";
import { useNavigate } from "react-router-dom";

const AuthContext = createContext<Context>({
    user: null,
    token: '',
    login: () => { },
    logout: () => { }
});

const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<string | null>(getUser() ?? null);
    const [token, setToken] = useState(getToken() ?? '');
    const navigate = useNavigate();
    const loginUrl = `${import.meta.env.VITE_API_URL}/api/auth/token`;

    const login = async (username: string, password: string) => {
        const formData = new FormData()
        formData.append('username', username);
        formData.append('password', password);

        const res = await fetch(loginUrl, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        setUser(username)
        setToken(data.access_token)
        saveUser(username);
        saveToken(data.access_token);
        navigate('/');
    }

    const logout = () => {
        setUser(null)
        setToken('')
        cleanToken();
        cleanUser();
    }

    return (
        <AuthContext.Provider value={{ user, token, login, logout }}>
            {children}
        </AuthContext.Provider>
    )
}

export default AuthProvider;

export const useAuth = () => {
    return useContext(AuthContext)
}