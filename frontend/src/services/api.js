import * as SecureStore from 'expo-secure-store';
const API_URL = 'http://127.0.0.1:5000';
const TOKEN_KEY = 'user_jwt_token';

export const getUserProfile = async() => {
    try {
        const token = await getToken();
        if (!token) return null;

        const response = await fetch(`${API_URL}/protected`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/JSON',
                'Authorization': `Bearer ${token}`
            }
        });

        return await response.json();
    } catch(error) {
        console.error("Profile fetch error:", error);
        return null;
    }
};

export const saveToken = async (token) => {
  try {
    await SecureStore.setItemAsync(TOKEN_KEY, token);
  } catch (error) {
    console.error("Error saving the token", error);
  }
};

export const getToken = async () => {
  try {
    return await SecureStore.getItemAsync(TOKEN_KEY);
  } catch (error) {
    console.error("Error retrieving the token", error);
    return null;
  }
};

export const deleteToken = async () => {
  try {
    await SecureStore.deleteItemAsync(TOKEN_KEY);
  } catch (error) {
    console.error("Error deleting the token", error);
  }
};

export const registerUser = async (username, email, password) => {
  try {
    const response = await fetch(`${API_URL}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    return await response.json();
  } catch (error) {
    console.error("Registration error:", error);
    return { message: "Network error. Is Flask running?" };
  }
};

export const loginUser = async (email, password) => {
  try {
    const response = await fetch(`${API_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    return await response.json();
  } catch (error) {
    console.error("Login error:", error);
    return { message: "Network error." };
  }
};