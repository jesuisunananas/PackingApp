import React from 'react';
import { 
  StyleSheet, View, Text, ScrollView, Image, 
  TouchableOpacity, SafeAreaView, Dimensions 
} from 'react-native';
import { 
  Camera, ChevronRight, Plus, Package, 
  LayoutGrid, Home, Box, FileText, User 
} from 'lucide-react-native';

const { width } = Dimensions.get('window');

export default function HomeScreen({ user, onNavigate, onLogout}) {
  // Mock data to match your Figma screenshots
  const stats = { items: 4, value: "2,495", newCount: 3 };
  
  const galleries = [
    { id: 1, title: 'Living Room Scan', count: 12, date: 'Feb 1, 2026', image: 'https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?q=80&w=400' },
    { id: 2, title: 'Master Bedroom', count: 24, date: 'Feb 3, 2026', image: 'https://images.unsplash.com/photo-1595526114035-0d45ed16cfbf?q=80&w=400' },
    { id: 3, title: 'Kitchen & Pantry', count: 45, date: 'Feb 5, 2026', image: 'https://images.unsplash.com/photo-1556911220-e15b29be8c8f?q=80&w=400' },
  ];

  return (
    <View style={styles.mainContainer}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
          
          {/* Header */}
          <View style={styles.header}>
            <View>
              <Text style={styles.welcomeText}>Hello, {user}</Text>
              <Text style={styles.subWelcomeText}>Ready to inventory your space?</Text>
            </View>
            <TouchableOpacity style={styles.profileButton} onPress={onLogout}>
              <User size={24} color="#9ca3af" /> 
            </TouchableOpacity>
          </View>

          {/* Main CTA: Video Capture */}
          <TouchableOpacity style={styles.ctaCard} activeOpacity={0.9}>
            <View style={styles.ctaIconBg}>
              <Camera size={24} color="white" />
            </View>
            <View>
              <Text style={styles.ctaTitle}>New Video Capture</Text>
              <Text style={styles.ctaSubtitle}>Scan a room to auto-detect items</Text>
            </View>
            <Camera size={120} color="rgba(255,255,255,0.1)" style={styles.ctaWatermark} />
          </TouchableOpacity>

          {/* Stats Row */}
          <View style={styles.statsRow}>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>TOTAL ITEMS</Text>
              <View style={styles.statValueRow}>
                <Text style={styles.statNumber}>{stats.items}</Text>
                <Text style={styles.statGreen}>+{stats.newCount} new</Text>
              </View>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>TOTAL VALUE</Text>
              <Text style={styles.statNumber}>${stats.value}</Text>
            </View>
          </View>

          {/* Galleries Section */}
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>My Galleries</Text>
            <TouchableOpacity style={styles.seeAllBtn}>
              <Text style={styles.seeAllText}>See all</Text>
              <ChevronRight size={16} color="#2563eb" />
            </TouchableOpacity>
          </View>
          
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.galleryScroll}>
            {galleries.map(item => (
              <TouchableOpacity key={item.id} style={styles.galleryCard}>
                <Image source={{ uri: item.image }} style={styles.galleryThumb} />
                <Text style={styles.galleryTitle}>{item.title}</Text>
                <Text style={styles.galleryMeta}>{item.count} items • {item.date}</Text>
              </TouchableOpacity>
            ))}
            <TouchableOpacity style={styles.addGalleryCard}>
              <Plus size={24} color="#9ca3af" />
              <Text style={styles.addGalleryText}>Add Room</Text>
            </TouchableOpacity>
          </ScrollView>

          {/* Saved Lists Section */}
          <View style={[styles.sectionHeader, { marginTop: 20 }]}>
            <Text style={styles.sectionTitle}>Saved Lists</Text>
            <TouchableOpacity style={styles.seeAllBtn}>
              <Text style={styles.seeAllText}>See all</Text>
              <ChevronRight size={16} color="#2563eb" />
            </TouchableOpacity>
          </View>

          <View style={styles.listContainer}>
            <TouchableOpacity style={styles.listItem}>
              <View style={[styles.listIcon, { backgroundColor: '#fff7ed' }]}>
                <Package size={20} color="#ea580c" />
              </View>
              <View style={styles.listText}>
                <Text style={styles.listTitle}>Donation Items</Text>
                <Text style={styles.listMeta}>12 items tracked</Text>
              </View>
              <ChevronRight size={18} color="#d1d5db" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.listItem}>
              <View style={[styles.listIcon, { backgroundColor: '#eff6ff' }]}>
                <LayoutGrid size={20} color="#2563eb" />
              </View>
              <View style={styles.listText}>
                <Text style={styles.listTitle}>Electronics Inventory</Text>
                <Text style={styles.listMeta}>8 items tracked</Text>
              </View>
              <ChevronRight size={18} color="#d1d5db" />
            </TouchableOpacity>
          </View>
        </ScrollView>
      </SafeAreaView>

      {/* Fixed Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Home size={24} color="#2563eb" />
          <Text style={[styles.navText, { color: '#2563eb' }]}>Home</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Box size={24} color="#9ca3af" />
          <Text style={styles.navText}>Items</Text>
        </TouchableOpacity>
        <View style={styles.cameraNavContainer}>
          <TouchableOpacity style={styles.cameraNavBtn}>
            <Camera size={28} color="white" />
          </TouchableOpacity>
        </View>
        <TouchableOpacity style={styles.navItem}>
          <FileText size={24} color="#9ca3af" />
          <Text style={styles.navText}>Reports</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={onLogout}>
          <User size={24} color="#9ca3af" />
          <Text style={styles.navText}>Profile</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  mainContainer: { flex: 1, backgroundColor: '#fcfcfc' },
  safeArea: { flex: 1 },
  scrollContent: { paddingHorizontal: 20, paddingTop: 10, paddingBottom: 120 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 25 },
  welcomeText: { fontSize: 24, fontWeight: 'bold', color: '#111827' },
  subWelcomeText: { fontSize: 14, color: '#6b7280', marginTop: 2 },
  profileButton: { 
    width: 44, height: 44, borderRadius: 22, 
    backgroundColor: '#f3f4f6', // Light gray background
    alignItems: 'center', justifyContent: 'center', 
    borderWidth: 2, borderColor: '#fff', 
    elevation: 3, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 4 
  },
  avatar: { width: '100%', height: '100%', borderRadius: 22 },
  ctaCard: { height: 180, backgroundColor: '#2563eb', borderRadius: 24, p: 24, padding: 24, justifyContent: 'flex-end', overflow: 'hidden', elevation: 8, shadowColor: '#2563eb', shadowOpacity: 0.3, shadowRadius: 10 },
  ctaIconBg: { width: 48, height: 48, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 24, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  ctaTitle: { color: 'white', fontSize: 20, fontWeight: 'bold' },
  ctaSubtitle: { color: 'rgba(255,255,255,0.8)', fontSize: 13, marginTop: 4 },
  ctaWatermark: { position: 'absolute', top: -20, right: -20 },
  statsRow: { flexDirection: 'row', gap: 12, marginTop: 20, marginBottom: 25 },
  statBox: { flex: 1, backgroundColor: 'white', padding: 16, borderRadius: 20, borderWeight: 1, borderColor: '#f3f4f6', borderWidth: 1 },
  statLabel: { fontSize: 10, fontWeight: 'bold', color: '#9ca3af', letterSpacing: 1 },
  statValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 6 },
  statNumber: { fontSize: 24, fontWeight: 'bold', color: '#111827', marginTop: 4 },
  statGreen: { fontSize: 12, fontWeight: 'bold', color: '#22c55e' },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  sectionTitle: { fontSize: 18, fontWeight: 'bold', color: '#111827' },
  seeAllBtn: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  seeAllText: { fontSize: 14, fontWeight: 'bold', color: '#2563eb' },
  galleryScroll: { marginHorizontal: -20, paddingHorizontal: 20 },
  galleryCard: { width: 160, marginRight: 16 },
  galleryThumb: { width: 160, height: 120, borderRadius: 20, marginBottom: 8 },
  galleryTitle: { fontSize: 14, fontWeight: 'bold', color: '#111827' },
  galleryMeta: { fontSize: 12, color: '#6b7280', marginTop: 2 },
  addGalleryCard: { width: 160, height: 120, borderRadius: 20, borderStyle: 'dashed', borderWidth: 2, borderColor: '#e5e7eb', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f9fafb' },
  addGalleryText: { fontSize: 12, fontWeight: 'bold', color: '#9ca3af', marginTop: 4 },
  listContainer: { gap: 12 },
  listItem: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'white', padding: 12, borderRadius: 20, borderWidth: 1, borderColor: '#f3f4f6' },
  listIcon: { width: 48, height: 48, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  listText: { flex: 1, marginLeft: 12 },
  listTitle: { fontSize: 14, fontWeight: 'bold', color: '#111827' },
  listMeta: { fontSize: 12, color: '#6b7280', marginTop: 2 },
  bottomNav: { position: 'absolute', bottom: 0, width: width, height: 90, backgroundColor: 'white', borderTopWidth: 1, borderTopColor: '#f3f4f6', flexDirection: 'row', paddingHorizontal: 15, paddingBottom: 20 },
  navItem: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  navText: { fontSize: 10, marginTop: 4, color: '#9ca3af', fontWeight: 'bold' },
  cameraNavContainer: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  cameraNavBtn: { width: 60, height: 60, backgroundColor: '#2563eb', borderRadius: 30, marginTop: -40, alignItems: 'center', justifyContent: 'center', elevation: 10, shadowColor: '#2563eb', shadowOpacity: 0.4, shadowRadius: 8 },
});