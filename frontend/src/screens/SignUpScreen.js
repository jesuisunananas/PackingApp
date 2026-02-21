import React, { useState } from 'react';
import { 
  StyleSheet, View, Text, TextInput, TouchableOpacity, 
  KeyboardAvoidingView, Platform, SafeAreaView 
} from 'react-native';
import { Mail, Lock, User, ArrowLeft, ShieldCheck, UserPlus } from 'lucide-react-native';

export default function SignUpScreen({ onBack, onSignUp }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <SafeAreaView style={styles.container}>
      <TouchableOpacity style={styles.backButton} onPress={onBack}>
        <ArrowLeft size={24} color="#374151" />
      </TouchableOpacity>

      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'} 
        style={styles.flex}
      >
        <View style={styles.innerContainer}>
          <View style={styles.header}>
            <View style={styles.logoBox}>
              <ShieldCheck size={40} color="white" />
            </View>
            <Text style={styles.title}>Create Account</Text>
            <Text style={styles.subtitle}>Join SmartScan to start optimizing your inventory</Text>
          </View>

          <View style={styles.form}>
            <Text style={styles.label}>Full Name</Text>
            <View style={styles.inputContainer}>
              <User size={20} color="#9ca3af" style={styles.icon} />
              <TextInput 
                style={styles.input}
                placeholder="John Doe"
                value={name}
                onChangeText={setName}
              />
            </View>

            <Text style={styles.label}>Email Address</Text>
            <View style={styles.inputContainer}>
              <Mail size={20} color="#9ca3af" style={styles.icon} />
              <TextInput 
                style={styles.input}
                placeholder="name@example.com"
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
              />
            </View>

            <Text style={styles.label}>Password</Text>
            <View style={styles.inputContainer}>
              <Lock size={20} color="#9ca3af" style={styles.icon} />
              <TextInput 
                style={styles.input}
                placeholder="••••••••"
                value={password}
                onChangeText={setPassword}
                secureTextEntry
              />
            </View>

            <TouchableOpacity 
              style={styles.signUpBtn} 
              onPress={() => onSignUp(name, email, password)}
            >
              <Text style={styles.signUpText}>Create Account</Text>
              <UserPlus size={20} color="white" />
            </TouchableOpacity>
          </View>

          <View style={styles.footer}>
            <Text style={styles.loginText}>
              Already have an account? <Text style={styles.linkText} onPress={onBack}>Sign In</Text>
            </Text>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  flex: { flex: 1 },
  backButton: {
    position: 'absolute',
    top: 60,
    left: 20,
    zIndex: 10,
  },
  innerContainer: { flex: 1, paddingHorizontal: 32 },
  header: { 
    alignItems: 'center', 
    marginTop: 80, 
    marginBottom: 40 
  },
  logoBox: { 
    width: 80, height: 80, backgroundColor: '#2563eb', 
    borderRadius: 20, alignItems: 'center', justifyContent: 'center',
    shadowColor: '#2563eb', shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2, shadowRadius: 15, elevation: 5
  },
  title: { fontSize: 30, fontWeight: 'bold', color: '#111827', marginTop: 24 },
  subtitle: { color: '#6b7280', marginTop: 8, textAlign: 'center' },
  form: { width: '100%' },
  label: { fontSize: 14, fontWeight: '600', color: '#374151', marginBottom: 8, marginLeft: 4 },
  inputContainer: { 
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#f9fafb',
    borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 12, height: 56, marginBottom: 16
  },
  icon: { marginLeft: 16 },
  input: { flex: 1, height: '100%', paddingHorizontal: 12, color: '#111827' },
  signUpBtn: { 
    backgroundColor: '#2563eb', height: 56, borderRadius: 12, 
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, marginTop: 10
  },
  signUpText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  footer: { marginTop: 32, alignItems: 'center' },
  loginText: { color: '#6b7280' },
  linkText: { color: '#2563eb', fontWeight: 'bold' },
});