import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HowItWorks from "./Components/HowItWorks/HowItWorks";
import Features from "./Components/Features/Features";
import Header from "./Components/Header/Header";
import Footer from "./Components/Lastfooter/Footer";
import Navbar from "./Components/Navbar/Navbar";
import Testimonials from "./Components/Testomonial/Testimonials";
import FAQ from "./Components/FAQ/FAQ";
import MedicalChatbot from "./Components/ChatbotUI/MedicalChatbot";

function App() {
  return (
    <Router>
      <Routes>
        {/* ✅ Home Page */}
        <Route
          path="/"
          element={
            <div className="bg-white min-h-screen">
              <Navbar />
              <Header />
              <Features />
              <HowItWorks />
              <Testimonials />
              <FAQ />
              <Footer />
            </div>
          }
        />

        {/* ✅ Chat Page */}
        <Route path="/chat" element={<MedicalChatbot />} />
      </Routes>
    </Router>
  );
}

export default App;
