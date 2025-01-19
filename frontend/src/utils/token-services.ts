const saveToken = (accessToken: string) => {
    localStorage.setItem('accessToken', accessToken)
}

const saveUser = (username: string) => {
    localStorage.setItem('user', username)
}

const cleanToken = () => {
    localStorage.removeItem('accessToken')
}

const cleanUser = () => {
    localStorage.removeItem('user')
}

const getToken = () => {
    return localStorage.getItem('accessToken')
}

const getUser = () => {
    return localStorage.getItem('user')
}

export { saveToken, cleanToken, getToken, getUser, saveUser, cleanUser }