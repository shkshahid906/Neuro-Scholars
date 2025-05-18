import { initializeApp } from "https://www.gstatic.com/firebasejs/11.7.3/firebase-app.js";
import { getDatabase, ref, onValue } from "https://www.gstatic.com/firebasejs/11.7.3/firebase-database.js";

// Your Firebase config (replace with your actual config)
const firebaseConfig = {
  apiKey: "AIzaSyALwKMLL2EVEjgNOBvpsF1Egi2s-9tYDgc",
  authDomain: "neuroscholars-be7d1.firebaseapp.com",
databaseURL: "https://neuroscholars-be7d1-default-rtdb.firebaseio.com",
  projectId: "neuroscholars-be7d1",
  storageBucket: "neuroscholars-be7d1.appspot.com",
  messagingSenderId: "1030438938151",
  appId: "1:1030438938151:web:7ca72fb446cec4535b9002",
  measurementId: "G-EZN6T048HV"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// Reference to soil sensor data node
const soilRef = ref(database, 'sensorData');

onValue(soilRef, (snapshot) => {
  const data = snapshot.val();

  if (data) {
    // Get the latest entry by timestamp (assuming timestamp field exists)
    const entries = Object.values(data);
    entries.sort((a, b) => b.timestamp - a.timestamp); // Descending by timestamp
    const latest = entries[0];

    // Update UI elements if data exists
    if (latest) {
      document.getElementById('soilN').textContent = latest.nitrogen !== undefined ? latest.nitrogen : '--';
      document.getElementById('soilP').textContent = latest.phosphorus !== undefined ? latest.phosphorus : '--';
      document.getElementById('soilK').textContent = latest.potassium !== undefined ? latest.potassium : '--';
      document.getElementById('soilPH').textContent = latest.ph !== undefined ? latest.ph : '--';
    }
  }
});
