import React, { useState } from 'react';
import { 
  StyleSheet, View, Text, TextInput, TouchableOpacity, 
  KeyboardAvoidingView, Platform, SafeAreaView 
} from 'react-native';
import { Mail, ArrowLeft, ShieldCheck, Send } from 'lucide-react-native';

export default function ForgotPasswordScreen({ onBack, onReset }) {
  const [email, setEmail] = useState('');

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
            <Text style={styles.title}>Reset Password</Text>
            <Text style={styles.subtitle}>
              Enter your email and we'll send you a link to reset your account.
            </Text>
          </View>

          <View style={styles.form}>
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

            <TouchableOpacity 
              style={styles.resetBtn} 
              onPress={() => onReset(email)}
            >
              <Text style={styles.resetText}>Send Reset Link</Text>
              <Send size={20} color="white" />
            </TouchableOpacity>
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
    top:60,
    left:20,
    // padding: 20,
    zindex:10,
    marginTop: Platform.OS === 'android' ? 30 : 0,
  },
  innerContainer: { 
    flex: 1, 
    paddingHorizontal: 32, 
    paddingBottom: 80, // Offset for better centering
    // justifyContent: 'center' 
  },
  header: { alignItems: 'center', marginTop:80, marginBottom: 48 },
  logoBox: { 
    width: 80, height: 80, backgroundColor: '#2563eb', 
    borderRadius: 20, alignItems: 'center', justifyContent: 'center',
    shadowColor: '#2563eb', shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2, shadowRadius: 15, elevation: 5
  },
  title: { fontSize: 30, fontWeight: 'bold', color: '#111827', marginTop: 24 },
  subtitle: { 
    color: '#6b7280', 
    marginTop: 12, 
    textAlign: 'center', 
    lineHeight: 20 
  },
  form: { width: '100%' },
  label: { fontSize: 14, fontWeight: '600', color: '#374151', marginBottom: 8, marginLeft: 4 },
  inputContainer: { 
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#f9fafb',
    borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 12, height: 56, marginBottom: 24
  },
  icon: { marginLeft: 16 },
  input: { flex: 1, height: '100%', paddingHorizontal: 12, color: '#111827' },
  resetBtn: { 
    backgroundColor: '#2563eb', height: 56, borderRadius: 12, 
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10,
    shadowColor: '#2563eb', shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3, shadowRadius: 8, elevation: 3
  },
  resetText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
});