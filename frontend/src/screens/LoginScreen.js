import React, { useState } from 'react';
import { 
  StyleSheet, View, Text, TextInput, TouchableOpacity, 
  KeyboardAvoidingView, Platform, 
  SafeAreaView // <--- Use the one from react-native!
} from 'react-native';

import { Mail, Lock, ArrowRight, ShieldCheck } from 'lucide-react-native';

export default function LoginScreen({ onLogin , onForgotPassword, onSignUp }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'} 
        style={styles.flex}
      >
        <View style={styles.innerContainer}>
          <View style={styles.header}>
            <View style={styles.logoBox}>
              <ShieldCheck size={40} color="white" />
            </View>
            <Text style={styles.title}>SmartScan</Text>
            <Text style={styles.subtitle}>AI-Powered Inventory Intelligence</Text>
          </View>

          <View style={styles.form}>
            <Text style={styles.label}>Email</Text>
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
                style={styles.forgotBtn}
                onPress={onForgotPassword}
            >
              <Text style={styles.forgotText}>Forgot password?</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.signInBtn} onPress={() => onLogin(email, password)}>
              <Text style={styles.signInText}>Sign In</Text>
              <ArrowRight size={20} color="white" />
            </TouchableOpacity>
          </View>

          <View style={styles.footer}>
            <Text style={styles.signUpText}>
              Don't have an account? <Text style={styles.linkText} onPress={onSignUp}>Sign Up</Text>
            </Text>
            <View style={styles.dividerContainer}>
              <View style={styles.line} />
              <Text style={styles.dividerText}>OR CONTINUE WITH</Text>
              <View style={styles.line} />
            </View>
            <View style={styles.socialRow}>
              <TouchableOpacity style={styles.socialBtn}><Text>Google</Text></TouchableOpacity>
              <TouchableOpacity style={styles.socialBtn}><Text>Apple</Text></TouchableOpacity>
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// THIS MUST BE IN THE FILE
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  flex: { flex: 1 },
//   innerContainer: { flex: 1, paddingHorizontal: 32, justifyContent: 'center' },  innerContainer: { flex: 1, paddingHorizontal: 32, justifyContent: 'center' },
  innerContainer: { flex: 1, paddingHorizontal: 32 },
  header: { alignItems: 'center', marginTop: 80, marginBottom: 48 },
  logoBox: { 
    width: 80, height: 80, backgroundColor: '#2563eb', 
    borderRadius: 20, alignItems: 'center', justifyContent: 'center',
    shadowColor: '#2563eb', shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2, shadowRadius: 15, elevation: 5
  },
  title: { fontSize: 30, fontWeight: 'bold', color: '#111827', marginTop: 24 },
  subtitle: { color: '#6b7280', marginTop: 8 },
  form: { width: '100%' },
  label: { fontSize: 14, fontWeight: '600', color: '#374151', marginBottom: 8, marginLeft: 4 },
  inputContainer: { 
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#f9fafb',
    borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 12, height: 56, marginBottom: 16
  },
  icon: { marginLeft: 16 },
  input: { flex: 1, height: '100%', paddingHorizontal: 12, color: '#111827' },
  forgotBtn: { alignSelf: 'flex-end', marginBottom: 24 },
  forgotText: { color: '#2563eb', fontWeight: '600', fontSize: 14 },
  signInBtn: { 
    backgroundColor: '#2563eb', height: 56, borderRadius: 12, 
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8
  },
  signInText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  footer: { marginTop: 40, alignItems: 'center', width: '100%' },
  signUpText: { color: '#6b7280' },
  linkText: { color: '#2563eb', fontWeight: 'bold' },
  dividerContainer: { flexDirection: 'row', alignItems: 'center', marginVertical: 32, width: '100%' },
  line: { flex: 1, height: 1, backgroundColor: '#e5e7eb' },
  dividerText: { marginHorizontal: 16, fontSize: 10, fontWeight: 'bold', color: '#9ca3af' },
  socialRow: { flexDirection: 'row', gap: 16, width: '100%' },
  socialBtn: { 
    flex: 1, height: 48, borderWidth: 1, borderColor: '#e5e7eb', 
    borderRadius: 12, alignItems: 'center', justifyContent: 'center' 
  }
});