import React, { useState, useEffect} from 'react';
import { Alert, View, ActivityIndicator } from 'react-native';
import LoginScreen from './src/screens/LoginScreen';
import ForgotPasswordScreen from './src/screens/ForgotPasswordScreen';
import SignUpScreen from './src/screens/SignUpScreen';
import HomeScreen from './src/screens/HomeScreen';
import { registerUser, loginUser, saveToken, getToken, deleteToken, getUserProfile } from './src/services/api'; 

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('login');
  const [isLoading, setIsLoading] = useState(true);
  const [userName, setUserName] = useState('')

  useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        const token = await getToken();
        if (token) {
          const profile = await getUserProfile();
          if (profile && profile.logged_in_as) {
            setUserName(profile.logged_in_as);
            setCurrentScreen('home'); 
          } else {
            await deleteToken();
          }
        }
      } catch (error) {
        console.error("Error checking token", error);
      } finally {
        setIsLoading(false);
      }
    };
    checkLoginStatus();
  }, []);

  const handleSignUp = async (name, email, pass) => {
    const result = await registerUser(name, email, pass);
    
    if (result.message === "User registered successfully") {
      console.log("Registration successful");
      setCurrentScreen('login');
    } else {
      Alert.alert("Error", result.message || "Sign up failed");
    }
  };

  const handleLogin = async (email, pass) => {
    const result = await loginUser(email, pass);
    if (result.access_token) {
      await saveToken(result.access_token);
      const profile = await getUserProfile();
      if (profile && profile.logged_in_as) {
        setUserName(profile.logged_in_as)
      }
      setCurrentScreen('home')
    } else {
      Alert.alert("Login Failed", result.message || "Invalid credentials");
    }
  };

  const handleLogout = async () => {
    Alert.alert("Log Out", "Are you sure you want to log out?", [
      { text: "Cancel", style: "cancel" },
      { 
        text: "Log Out", 
        style: "destructive",
        onPress: async () => {
          await deleteToken(); // Destroy the token
          setUserName('');     // Clear the name
          setCurrentScreen('login'); // Send back to login
        }
      }
    ]);
  };

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#2563eb" />
      </View>
    );
  }

  // --- Navigation Routing ---
  if (currentScreen === 'forgot-password') {
    return (
      <ForgotPasswordScreen 
        onBack={() => setCurrentScreen('login')} 
        onReset={(email) => console.log('Resetting:', email)}
      />
    );
  }

  if (currentScreen === 'signup') {
    return (
      <SignUpScreen 
        onBack={() => setCurrentScreen('login')} 
        onSignUp={handleSignUp} 
      />
    );
  }

  if (currentScreen === 'home') {
    return (
      <HomeScreen 
        user={userName || "User"}
        onNavigate={(screen) => setCurrentScreen(screen)}
        onLogout={handleLogout}
      />
    );
  }

  return (
    <LoginScreen 
      onLogin={handleLogin} 
      onForgotPassword={() => setCurrentScreen('forgot-password')} 
      onSignUp={() => setCurrentScreen('signup')}
    />
  );
}

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: '#fff',
//     alignItems: 'center',
//     justifyContent: 'center',
//   },
// });
