export interface Context {
    user: string | null;
    token: string;
    login: (username: string, password: string) => void;
    logout: () => void;
}

export interface ChatRecord {
    id?: number;
    role_type: string;
    content: string;
    created?: Date;
}